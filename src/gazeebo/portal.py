"""Compositor-neutral pointer control through XDG desktop portals."""

from __future__ import annotations

import asyncio
import secrets
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

from dbus_next.aio.message_bus import MessageBus
from dbus_next.constants import MessageFlag, MessageType
from dbus_next.message import Message
from dbus_next.signature import Variant

from gazeebo.contracts import DisplayRegion
from gazeebo.geometry import DisplayTopology

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

PORTAL_DESTINATION = "org.freedesktop.portal.Desktop"
PORTAL_PATH = "/org/freedesktop/portal/desktop"
REMOTE_DESKTOP = "org.freedesktop.portal.RemoteDesktop"
SCREEN_CAST = "org.freedesktop.portal.ScreenCast"
REQUEST = "org.freedesktop.portal.Request"
SESSION = "org.freedesktop.portal.Session"
POINTER_DEVICE = 2
MONITOR_SOURCE = 1
MULTIPLE_SOURCES = True
PAIR_LENGTH = 2


class PortalError(RuntimeError):
    """Desktop authorization or pointer delivery failed safely."""


class _Bus(Protocol):
    async def call(self, message: Message) -> Message | None:
        """Call one portal method."""

    def send(self, message: Message) -> asyncio.Future[None] | None:
        """Queue one message and return its transport-completion future."""

    def add_message_handler(self, handler: Callable[[Message], object]) -> None:
        """Register a signal handler."""

    def disconnect(self) -> None:
        """Disconnect from the user bus."""


@dataclass(frozen=True, slots=True)
class PortalStream:
    """The selected portal stream and its compositor-coordinate properties."""

    node_id: int
    properties: dict[str, object]


class _RequestBroker:
    """Match asynchronous portal request responses to method-return paths."""

    def __init__(self, bus: _Bus, timeout: float = 60.0) -> None:
        self._bus = bus
        self._timeout = timeout
        self._pending: dict[str, asyncio.Future[tuple[int, dict[str, object]]]] = {}
        self._early: dict[str, tuple[int, dict[str, object]]] = {}
        bus.add_message_handler(self._handle_message)

    def _handle_message(self, message: Message) -> bool:
        if (
            message.message_type is not MessageType.SIGNAL
            or message.interface != REQUEST
            or message.member != "Response"
            or message.path is None
        ):
            return False
        response = int(message.body[0])
        raw_results = cast("dict[str, object]", message.body[1])
        results = {key: _unwrap(value) for key, value in raw_results.items()}
        future = self._pending.pop(message.path, None)
        if future is not None and not future.done():
            future.set_result((response, results))
        else:
            self._early[message.path] = (response, results)
        return True

    async def request(
        self,
        interface: str,
        member: str,
        signature: str,
        body: list[object],
    ) -> dict[str, object]:
        """Call a portal request method and await its Response signal."""
        reply = await self._bus.call(
            Message(
                destination=PORTAL_DESTINATION,
                path=PORTAL_PATH,
                interface=interface,
                member=member,
                signature=signature,
                body=body,
            )
        )
        if reply is None or reply.message_type is MessageType.ERROR:
            detail = "portal method returned no reply"
            if reply is not None and reply.body:
                detail = str(reply.body[0])
            raise PortalError(detail)
        if not reply.body:
            msg = "portal request returned no request path"
            raise PortalError(msg)
        request_path = str(reply.body[0])
        early = self._early.pop(request_path, None)
        if early is None:
            future = asyncio.get_running_loop().create_future()
            self._pending[request_path] = future
            try:
                response, results = await asyncio.wait_for(future, self._timeout)
            except TimeoutError as error:
                self._pending.pop(request_path, None)
                msg = f"portal request timed out: {member}"
                raise PortalError(msg) from error
        else:
            response, results = early
        if response != 0:
            msg = f"portal request was denied or cancelled: {member}"
            raise PortalError(msg)
        return cast("dict[str, object]", results)


def _unwrap(value: object) -> object:
    """Remove one dbus-next Variant wrapper while retaining structured values."""
    if isinstance(value, Variant):
        return cast("object", value.value)
    return value


def _pair(value: object, name: str) -> tuple[int, int]:
    """Validate a two-integer stream geometry property."""
    unwrapped = _unwrap(value)
    if not isinstance(unwrapped, (list, tuple)) or len(unwrapped) != PAIR_LENGTH:
        msg = f"portal stream has invalid {name} geometry"
        raise PortalError(msg)
    return int(unwrapped[0]), int(unwrapped[1])


def parse_portal_streams(raw_streams: object) -> tuple[PortalStream, ...]:
    """Validate the Start response's stream list."""
    streams = _unwrap(raw_streams)
    if not isinstance(streams, (list, tuple)) or not streams:
        msg = "portal returned no authorized display streams"
        raise PortalError(msg)
    parsed: list[PortalStream] = []
    for raw_stream in streams:
        if not isinstance(raw_stream, (list, tuple)) or len(raw_stream) != PAIR_LENGTH:
            msg = "portal returned a malformed display stream"
            raise PortalError(msg)
        node_id = int(raw_stream[0])
        raw_properties = raw_stream[1]
        if not isinstance(raw_properties, dict):
            msg = "portal stream properties are malformed"
            raise PortalError(msg)
        properties = {str(key): _unwrap(value) for key, value in raw_properties.items()}
        parsed.append(PortalStream(node_id, properties))
    return tuple(parsed)


def regions_from_streams(
    streams: Sequence[PortalStream],
) -> tuple[tuple[DisplayRegion, ...], dict[str, int]]:
    """Build all authorized logical regions and their stream-node mapping."""
    regions: list[DisplayRegion] = []
    nodes: dict[str, int] = {}
    for stream in streams:
        properties = stream.properties
        size_value = properties.get("logical_size", properties.get("size"))
        if size_value is None:
            msg = "portal stream omits logical size"
            raise PortalError(msg)
        width, height = _pair(size_value, "size")
        position_value = properties.get("position")
        if position_value is None:
            if len(streams) > 1:
                msg = "portal streams omit required multi-display positions"
                raise PortalError(msg)
            x, y = 0, 0
        else:
            x, y = _pair(position_value, "position")
        mapping = properties.get("mapping_id")
        region_id = str(mapping) if mapping else f"stream-{stream.node_id}"
        if region_id in nodes:
            region_id = f"{region_id}-{stream.node_id}"
        region = DisplayRegion(region_id, x, y, width, height)
        regions.append(region)
        nodes[region_id] = stream.node_id
    return tuple(regions), nodes


class PortalPointerController:
    """Own one non-persistent RemoteDesktop pointer session."""

    def __init__(
        self,
        bus: _Bus,
        session_path: str,
        streams: Sequence[PortalStream],
    ) -> None:
        """Build immutable authorized-display geometry from a started session."""
        self._bus = bus
        self._session_path = session_path
        self._regions, self._nodes = regions_from_streams(streams)
        self._topology_id = DisplayTopology(self._regions).topology_id
        self._pending_sends: set[asyncio.Future[None]] = set()
        self._closed = False
        self._disconnected = False
        bus.add_message_handler(self._handle_session_signal)

    @classmethod
    async def authorize(cls, *, request_timeout: float = 300.0) -> PortalPointerController:
        """Request non-persistent pointer and multi-monitor authorization."""
        connected = await MessageBus().connect()
        bus = cast("_Bus", connected)
        broker = _RequestBroker(bus, request_timeout)
        session_token = _token("session")
        create_results = await broker.request(
            REMOTE_DESKTOP,
            "CreateSession",
            "a{sv}",
            [
                {
                    "handle_token": Variant("s", _token("request")),
                    "session_handle_token": Variant("s", session_token),
                }
            ],
        )
        session_path = create_results.get("session_handle")
        if not isinstance(session_path, str):
            bus.disconnect()
            msg = "portal did not create a session"
            raise PortalError(msg)
        try:
            await broker.request(
                SCREEN_CAST,
                "SelectSources",
                "oa{sv}",
                [
                    session_path,
                    {
                        "handle_token": Variant("s", _token("request")),
                        "types": Variant("u", MONITOR_SOURCE),
                        "multiple": Variant("b", MULTIPLE_SOURCES),
                    },
                ],
            )
            await broker.request(
                REMOTE_DESKTOP,
                "SelectDevices",
                "oa{sv}",
                [
                    session_path,
                    {
                        "handle_token": Variant("s", _token("request")),
                        "types": Variant("u", POINTER_DEVICE),
                        "persist_mode": Variant("u", 0),
                    },
                ],
            )
            start = await broker.request(
                REMOTE_DESKTOP,
                "Start",
                "osa{sv}",
                [session_path, "", {"handle_token": Variant("s", _token("request"))}],
            )
            raw_devices = _parse_devices(start.get("devices", 0))
            _require_pointer_authorization(raw_devices)
            streams = parse_portal_streams(start.get("streams"))
            return cls(bus, session_path, streams)
        except BaseException:
            pending = _close_session(bus, session_path)
            if pending is not None:
                await asyncio.gather(pending, return_exceptions=True)
            bus.disconnect()
            raise

    @property
    def regions(self) -> tuple[DisplayRegion, ...]:
        """Return immutable authorized display geometry."""
        return self._regions

    @property
    def topology_id(self) -> str:
        """Return an identity for the authorized session geometry."""
        return self._topology_id

    @property
    def closed(self) -> bool:
        """Return whether the desktop invalidated or the app closed the session."""
        return self._closed

    def move(self, region_id: str, x: float, y: float) -> None:
        """Send absolute motion within one authorized stream."""
        if self._closed:
            msg = "portal pointer session is closed"
            raise PortalError(msg)
        region = next((item for item in self._regions if item.region_id == region_id), None)
        if region is None or not 0 <= x < region.width or not 0 <= y < region.height:
            msg = "pointer target is outside authorized portal geometry"
            raise PortalError(msg)
        self._send(
            "NotifyPointerMotionAbsolute",
            "oa{sv}udd",
            [self._session_path, {}, self._nodes[region_id], x, y],
        )

    def _send(self, member: str, signature: str, body: list[object]) -> None:
        self._queue_message(
            Message(
                destination=PORTAL_DESTINATION,
                path=PORTAL_PATH,
                interface=REMOTE_DESKTOP,
                member=member,
                signature=signature,
                body=body,
            )
        )

    def _queue_message(self, message: Message) -> None:
        pending = self._bus.send(message)
        if pending is not None:
            self._pending_sends.add(pending)
            pending.add_done_callback(self._pending_sends.discard)

    def _handle_session_signal(self, message: Message) -> bool:
        if (
            message.message_type is MessageType.SIGNAL
            and message.interface == SESSION
            and message.member == "Closed"
            and message.path == self._session_path
        ):
            self._closed = True
            return True
        return False

    async def close(self) -> None:
        """Drain motion messages and close the session idempotently."""
        if not self._closed:
            self._closed = True
            pending = _close_session(self._bus, self._session_path)
            if pending is not None:
                self._pending_sends.add(pending)
                pending.add_done_callback(self._pending_sends.discard)
        if self._pending_sends:
            await asyncio.gather(*tuple(self._pending_sends), return_exceptions=True)
        if not self._disconnected:
            self._disconnected = True
            self._bus.disconnect()


def _parse_devices(value: object) -> int:
    if not isinstance(value, int):
        msg = "portal returned malformed device authorization"
        raise PortalError(msg)
    return value


def _require_pointer_authorization(devices: int) -> None:
    if devices & POINTER_DEVICE == 0:
        msg = "portal session lacks pointer authorization"
        raise PortalError(msg)


def _close_session(
    bus: _Bus,
    session_path: str,
) -> asyncio.Future[None] | None:
    return bus.send(
        Message(
            destination=PORTAL_DESTINATION,
            path=session_path,
            interface=SESSION,
            member="Close",
            flags=MessageFlag.NO_REPLY_EXPECTED,
        )
    )


def _token(prefix: str) -> str:
    return f"gazeebo_{prefix}_{secrets.token_hex(8)}"
