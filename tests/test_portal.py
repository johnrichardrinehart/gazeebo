"""Tests for compositor-neutral portal geometry and pointer messages."""

from __future__ import annotations

import asyncio
import unittest
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from dbus_next.constants import MessageFlag, MessageType
from dbus_next.message import Message
from dbus_next.signature import Variant

from gazeebo.portal import (
    MULTIPLE_SOURCES,
    POINTER_DEVICE,
    SESSION,
    PortalError,
    PortalPointerController,
    PortalStream,
    parse_portal_streams,
    regions_from_streams,
)

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(slots=True)
class StubBus:
    """Record D-Bus messages and expose registered signal handlers."""

    sent: list[Message] = field(default_factory=list)
    handlers: list[Callable[[Message], object]] = field(default_factory=list)
    disconnected: bool = False

    async def call(self, message: Message) -> Message | None:
        """Reject unexpected request calls in controller-only tests."""
        del message
        msg = "unexpected portal call"
        raise AssertionError(msg)

    def send(self, message: Message) -> asyncio.Future[None] | None:
        """Record one outgoing event."""
        self.sent.append(message)
        return None

    def add_message_handler(self, handler: Callable[[Message], object]) -> None:
        """Record a signal handler."""
        self.handlers.append(handler)

    def disconnect(self) -> None:
        """Record connection cleanup."""
        self.disconnected = True


class PortalTests(unittest.TestCase):
    """Lock stream parsing, least-privilege messages, and cleanup."""

    def test_requests_pointer_device_bit(self) -> None:
        """RemoteDesktop device bit two is pointer access, not keyboard access."""
        assert POINTER_DEVICE == 2
        assert MULTIPLE_SOURCES

    def test_selected_stream_retains_compositor_geometry(self) -> None:
        """An authorized output retains its logical position and size."""
        raw = [
            [
                11,
                {
                    "position": Variant("(ii)", [3840, 360]),
                    "size": Variant("(ii)", [5120, 2880]),
                    "logical_size": Variant("(ii)", [2560, 1440]),
                    "mapping_id": Variant("s", "selected"),
                },
            ]
        ]
        streams = parse_portal_streams(Variant("a(ua{sv})", raw))
        regions, nodes = regions_from_streams(streams)
        assert len(regions) == 1
        assert regions[0].region_id == "selected"
        assert (regions[0].x, regions[0].y, regions[0].width, regions[0].height) == (
            3840,
            360,
            2560,
            1440,
        )
        assert nodes == {"selected": 11}

    def test_multiple_display_streams_retain_combined_geometry(self) -> None:
        """Every authorized stream becomes an addressable pointer region."""
        streams = (
            PortalStream(1, {"size": (800, 600), "position": (0, 0)}),
            PortalStream(2, {"size": (1000, 700), "position": (800, 100)}),
        )
        regions, nodes = regions_from_streams(streams)
        assert [(item.x, item.y, item.width, item.height) for item in regions] == [
            (0, 0, 800, 600),
            (800, 100, 1000, 700),
        ]
        assert nodes == {"stream-1": 1, "stream-2": 2}

    def test_multiple_streams_require_logical_positions(self) -> None:
        """Missing geometry cannot silently overlap authorized displays."""
        streams = (
            PortalStream(1, {"size": (800, 600)}),
            PortalStream(2, {"size": (1000, 700), "position": (800, 0)}),
        )
        with self.assertRaisesRegex(PortalError, "positions"):
            regions_from_streams(streams)

    def test_controller_sends_only_pointer_motion(self) -> None:
        """The backend targets an authorized stream without button messages."""
        bus = StubBus()
        controller = PortalPointerController(
            bus,
            "/session/fixture",
            (PortalStream(42, {"size": (1000, 700), "position": (0, 0)}),),
        )
        region_id = controller.regions[0].region_id
        controller.move(region_id, 500.0, 300.0)
        asyncio.run(controller.close())
        asyncio.run(controller.close())

        assert [message.member for message in bus.sent] == [
            "NotifyPointerMotionAbsolute",
            "Close",
        ]
        assert bus.sent[0].body[2:] == [42, 500.0, 300.0]
        assert not bus.sent[0].flags & MessageFlag.NO_REPLY_EXPECTED
        assert bus.disconnected

    def test_session_closed_signal_marks_topology_stale_and_disconnects(self) -> None:
        """Desktop invalidation prevents further events and still closes the bus."""
        bus = StubBus()
        controller = PortalPointerController(
            bus,
            "/session/fixture",
            (PortalStream(7, {"size": (640, 480), "position": (0, 0)}),),
        )
        closed = Message(
            message_type=MessageType.SIGNAL,
            path="/session/fixture",
            interface=SESSION,
            member="Closed",
        )
        assert any(bool(handler(closed)) for handler in bus.handlers)
        assert controller.closed
        with self.assertRaisesRegex(PortalError, "closed"):
            controller.move(controller.regions[0].region_id, 10.0, 10.0)
        asyncio.run(controller.close())
        assert bus.disconnected
        assert bus.sent == []

    def test_close_drains_motion_transport(self) -> None:
        """Termination drains queued motion before disconnecting the bus."""

        @dataclass(slots=True)
        class DeferredBus(StubBus):
            pending: list[asyncio.Future[None]] = field(default_factory=list)

            def send(self, message: Message) -> asyncio.Future[None]:
                self.sent.append(message)
                future = asyncio.get_running_loop().create_future()
                self.pending.append(future)
                asyncio.get_running_loop().call_soon(future.set_result, None)
                return future

            def disconnect(self) -> None:
                assert all(future.done() for future in self.pending)
                self.disconnected = True

        async def exercise() -> tuple[DeferredBus, PortalPointerController]:
            bus = DeferredBus()
            controller = PortalPointerController(
                bus,
                "/session/fixture",
                (PortalStream(42, {"size": (1000, 700), "position": (0, 0)}),),
            )
            controller.move(controller.regions[0].region_id, 10.0, 20.0)
            await controller.close()
            return bus, controller

        bus, controller = asyncio.run(exercise())
        assert controller.closed
        assert bus.disconnected
        assert [message.member for message in bus.sent] == [
            "NotifyPointerMotionAbsolute",
            "Close",
        ]


if __name__ == "__main__":
    unittest.main()
