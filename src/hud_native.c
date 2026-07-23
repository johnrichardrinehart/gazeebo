#define _POSIX_C_SOURCE 200809L

#include "wlr-layer-shell-unstable-v1-client-protocol.h"
#include "xdg-output-unstable-v1-client-protocol.h"

#include <cairo.h>
#include <errno.h>
#include <fcntl.h>
#include <pango/pangocairo.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <wayland-client.h>

#define HUD_WIDTH 540
#define HUD_HEIGHT 112
#define HUD_BUFFER_COUNT 3
#define HUD_MARGIN 16
#define HUD_RADIUS 9.0
#define HUD_MAX_OUTPUTS 32
#define GAME_NAMESPACE "gazeebo-calibration-game"

typedef struct HudState HudState;

typedef struct {
  struct wl_buffer *buffer;
  uint32_t *pixels;
  bool busy;
} HudBuffer;

typedef struct {
  struct wl_output *output;
  struct zxdg_output_v1 *xdg_output;
  int32_t x;
  int32_t y;
  int32_t width;
  int32_t height;
  char name[128];
  char description[384];
} HudOutput;

struct HudState {
  struct wl_display *display;
  struct wl_registry *registry;
  struct wl_compositor *compositor;
  struct wl_shm *shm;
  struct zwlr_layer_shell_v1 *layer_shell;
  struct zxdg_output_manager_v1 *output_manager;
  struct wl_shm_pool *pool;
  struct wl_surface *surface;
  struct zwlr_layer_surface_v1 *layer_surface;
  void *pool_data;
  size_t pool_size;
  int32_t frame_width;
  int32_t frame_height;
  HudBuffer buffers[HUD_BUFFER_COUNT];
  HudOutput outputs[HUD_MAX_OUTPUTS];
  size_t output_count;
  int next_buffer;
  bool configured;
  bool closed;
  bool failed;
  char error[256];
};

static void set_error(HudState *state, const char *message) {
  if (!state->failed) {
    snprintf(state->error, sizeof(state->error), "%s", message);
  }
  state->failed = true;
}

static void copy_error(const HudState *state, char *error, size_t error_size) {
  if (error != NULL && error_size > 0) {
    snprintf(error, error_size, "%s", state->error);
  }
}

static int anonymous_file(size_t size) {
  const char *runtime = getenv("XDG_RUNTIME_DIR");
  if (runtime == NULL || runtime[0] == '\0') {
    errno = ENOENT;
    return -1;
  }
  char path[4096];
  snprintf(path, sizeof(path), "%s/gazeebo-renderer-XXXXXX", runtime);
  int fd = mkstemp(path);
  if (fd < 0) {
    return -1;
  }
  unlink(path);
  int flags = fcntl(fd, F_GETFD);
  if (flags >= 0) {
    (void)fcntl(fd, F_SETFD, flags | FD_CLOEXEC);
  }
  if (ftruncate(fd, (off_t)size) != 0) {
    close(fd);
    return -1;
  }
  return fd;
}

static void buffer_release(void *data, struct wl_buffer *buffer) {
  (void)buffer;
  ((HudBuffer *)data)->busy = false;
}

static const struct wl_buffer_listener BUFFER_LISTENER = {
    .release = buffer_release,
};

static void output_logical_position(void *data,
                                    struct zxdg_output_v1 *xdg_output,
                                    int32_t x, int32_t y) {
  (void)xdg_output;
  HudOutput *output = data;
  output->x = x;
  output->y = y;
}

static void output_logical_size(void *data, struct zxdg_output_v1 *xdg_output,
                                int32_t width, int32_t height) {
  (void)xdg_output;
  HudOutput *output = data;
  output->width = width;
  output->height = height;
}

static void output_done(void *data, struct zxdg_output_v1 *xdg_output) {
  (void)data;
  (void)xdg_output;
}

static void output_name(void *data, struct zxdg_output_v1 *xdg_output,
                        const char *name) {
  (void)xdg_output;
  HudOutput *output = data;
  snprintf(output->name, sizeof(output->name), "%s", name);
}

static void output_description(void *data, struct zxdg_output_v1 *xdg_output,
                               const char *description) {
  (void)xdg_output;
  HudOutput *output = data;
  snprintf(output->description, sizeof(output->description), "%s", description);
}

static const struct zxdg_output_v1_listener OUTPUT_LISTENER = {
    .logical_position = output_logical_position,
    .logical_size = output_logical_size,
    .done = output_done,
    .name = output_name,
    .description = output_description,
};

static void registry_global(void *data, struct wl_registry *registry,
                            uint32_t name, const char *interface,
                            uint32_t version) {
  HudState *state = data;
  if (strcmp(interface, wl_compositor_interface.name) == 0) {
    uint32_t bind_version = version < 4 ? version : 4;
    state->compositor = wl_registry_bind(
        registry, name, &wl_compositor_interface, bind_version);
  } else if (strcmp(interface, wl_shm_interface.name) == 0) {
    state->shm = wl_registry_bind(registry, name, &wl_shm_interface, 1);
  } else if (strcmp(interface, zwlr_layer_shell_v1_interface.name) == 0) {
    uint32_t bind_version = version < 4 ? version : 4;
    state->layer_shell = wl_registry_bind(
        registry, name, &zwlr_layer_shell_v1_interface, bind_version);
  } else if (strcmp(interface, zxdg_output_manager_v1_interface.name) == 0) {
    uint32_t bind_version = version < 3 ? version : 3;
    state->output_manager = wl_registry_bind(
        registry, name, &zxdg_output_manager_v1_interface, bind_version);
  } else if (strcmp(interface, wl_output_interface.name) == 0 &&
             state->output_count < HUD_MAX_OUTPUTS) {
    HudOutput *output = &state->outputs[state->output_count++];
    uint32_t bind_version = version < 3 ? version : 3;
    output->output =
        wl_registry_bind(registry, name, &wl_output_interface, bind_version);
  }
}

static void registry_remove(void *data, struct wl_registry *registry,
                            uint32_t name) {
  (void)data;
  (void)registry;
  (void)name;
}

static const struct wl_registry_listener REGISTRY_LISTENER = {
    .global = registry_global,
    .global_remove = registry_remove,
};

static void layer_configure(void *data,
                            struct zwlr_layer_surface_v1 *layer_surface,
                            uint32_t serial, uint32_t width, uint32_t height) {
  (void)width;
  (void)height;
  HudState *state = data;
  zwlr_layer_surface_v1_ack_configure(layer_surface, serial);
  state->configured = true;
}

static void layer_closed(void *data,
                         struct zwlr_layer_surface_v1 *layer_surface) {
  (void)layer_surface;
  ((HudState *)data)->closed = true;
}

static const struct zwlr_layer_surface_v1_listener LAYER_LISTENER = {
    .configure = layer_configure,
    .closed = layer_closed,
};

static int create_buffers(HudState *state) {
  size_t frame_size = (size_t)state->frame_width * (size_t)state->frame_height *
                      sizeof(uint32_t);
  state->pool_size = HUD_BUFFER_COUNT * frame_size;
  int fd = anonymous_file(state->pool_size);
  if (fd < 0) {
    set_error(state, "cannot allocate Wayland renderer buffers");
    return -1;
  }
  state->pool_data =
      mmap(NULL, state->pool_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (state->pool_data == MAP_FAILED) {
    close(fd);
    state->pool_data = NULL;
    set_error(state, "cannot map Wayland renderer buffers");
    return -1;
  }
  state->pool = wl_shm_create_pool(state->shm, fd, (int32_t)state->pool_size);
  close(fd);
  for (int index = 0; index < HUD_BUFFER_COUNT; index++) {
    HudBuffer *frame = &state->buffers[index];
    frame->pixels = (uint32_t *)((unsigned char *)state->pool_data +
                                 (size_t)index * frame_size);
    frame->buffer = wl_shm_pool_create_buffer(
        state->pool, (int32_t)((size_t)index * frame_size), state->frame_width,
        state->frame_height, state->frame_width * 4, WL_SHM_FORMAT_ARGB8888);
    wl_buffer_add_listener(frame->buffer, &BUFFER_LISTENER, frame);
  }
  return 0;
}

static void rounded_rectangle(cairo_t *cairo, double x, double y, double width,
                              double height, double radius) {
  const double pi = 3.14159265358979323846;
  cairo_new_sub_path(cairo);
  cairo_arc(cairo, x + width - radius, y + radius, radius, -pi / 2.0, 0.0);
  cairo_arc(cairo, x + width - radius, y + height - radius, radius, 0.0,
            pi / 2.0);
  cairo_arc(cairo, x + radius, y + height - radius, radius, pi / 2.0, pi);
  cairo_arc(cairo, x + radius, y + radius, radius, pi, 3.0 * pi / 2.0);
  cairo_close_path(cairo);
}

static void render_hud(HudState *state, HudBuffer *frame, const char *text) {
  memset(frame->pixels, 0,
         (size_t)state->frame_width * (size_t)state->frame_height *
             sizeof(*frame->pixels));
  cairo_surface_t *surface = cairo_image_surface_create_for_data(
      (unsigned char *)frame->pixels, CAIRO_FORMAT_ARGB32, state->frame_width,
      state->frame_height, state->frame_width * 4);
  cairo_t *cairo = cairo_create(surface);

  cairo_set_operator(cairo, CAIRO_OPERATOR_SOURCE);
  cairo_set_source_rgba(cairo, 0.07, 0.07, 0.07, 0.74);
  rounded_rectangle(cairo, 1.0, 1.0, state->frame_width - 2.0,
                    state->frame_height - 2.0, HUD_RADIUS);
  cairo_fill_preserve(cairo);
  cairo_set_line_width(cairo, 1.0);
  cairo_set_source_rgba(cairo, 1.0, 1.0, 1.0, 0.25);
  cairo_stroke(cairo);

  PangoLayout *layout = pango_cairo_create_layout(cairo);
  PangoFontDescription *font =
      pango_font_description_from_string("Monospace 13");
  pango_layout_set_font_description(layout, font);
  pango_layout_set_text(layout, text, -1);
  pango_layout_set_spacing(layout, 2 * PANGO_SCALE);
  cairo_move_to(cairo, 14.0, 12.0);
  cairo_set_source_rgba(cairo, 1.0, 1.0, 1.0, 0.96);
  pango_cairo_show_layout(cairo, layout);

  pango_font_description_free(font);
  g_object_unref(layout);
  cairo_destroy(cairo);
  cairo_surface_flush(surface);
  cairo_surface_mark_dirty(surface);
  cairo_surface_destroy(surface);
}

static void render_game(HudState *state, HudBuffer *frame, double x, double y,
                        double diameter, const char *label) {
  memset(frame->pixels, 0,
         (size_t)state->frame_width * (size_t)state->frame_height *
             sizeof(*frame->pixels));
  cairo_surface_t *surface = cairo_image_surface_create_for_data(
      (unsigned char *)frame->pixels, CAIRO_FORMAT_ARGB32, state->frame_width,
      state->frame_height, state->frame_width * 4);
  cairo_t *cairo = cairo_create(surface);

  const double pi = 3.14159265358979323846;
  cairo_set_operator(cairo, CAIRO_OPERATOR_SOURCE);
  cairo_set_source_rgba(cairo, 0.92, 0.04, 0.04, 0.84);
  cairo_arc(cairo, x, y, diameter / 2.0, 0.0, 2.0 * pi);
  cairo_fill_preserve(cairo);
  cairo_set_line_width(cairo, 4.0);
  cairo_set_source_rgba(cairo, 1.0, 0.78, 0.78, 0.98);
  cairo_stroke(cairo);

  PangoLayout *layout = pango_cairo_create_layout(cairo);
  PangoFontDescription *font =
      pango_font_description_from_string("Sans Bold 16");
  pango_layout_set_font_description(layout, font);
  pango_layout_set_text(layout, label, -1);
  int text_width = 0;
  int text_height = 0;
  pango_layout_get_pixel_size(layout, &text_width, &text_height);
  cairo_set_source_rgba(cairo, 0.04, 0.04, 0.04, 0.78);
  rounded_rectangle(cairo, 18.0, 18.0, text_width + 24.0, text_height + 18.0,
                    HUD_RADIUS);
  cairo_fill(cairo);
  cairo_move_to(cairo, 30.0, 27.0);
  cairo_set_source_rgba(cairo, 1.0, 1.0, 1.0, 0.98);
  pango_cairo_show_layout(cairo, layout);

  pango_font_description_free(font);
  g_object_unref(layout);
  cairo_destroy(cairo);
  cairo_surface_flush(surface);
  cairo_surface_mark_dirty(surface);
  cairo_surface_destroy(surface);
}

static HudBuffer *available_buffer(HudState *state) {
  for (int attempt = 0; attempt < HUD_BUFFER_COUNT; attempt++) {
    int candidate = (state->next_buffer + attempt) % HUD_BUFFER_COUNT;
    if (!state->buffers[candidate].busy) {
      state->next_buffer = (candidate + 1) % HUD_BUFFER_COUNT;
      return &state->buffers[candidate];
    }
  }
  return NULL;
}

static void cleanup(HudState *state) {
  if (state == NULL) {
    return;
  }
  for (int index = 0; index < HUD_BUFFER_COUNT; index++) {
    if (state->buffers[index].buffer != NULL) {
      wl_buffer_destroy(state->buffers[index].buffer);
    }
  }
  for (size_t index = 0; index < state->output_count; index++) {
    if (state->outputs[index].xdg_output != NULL) {
      zxdg_output_v1_destroy(state->outputs[index].xdg_output);
    }
    if (state->outputs[index].output != NULL) {
      wl_output_destroy(state->outputs[index].output);
    }
  }
  if (state->layer_surface != NULL) {
    zwlr_layer_surface_v1_destroy(state->layer_surface);
  }
  if (state->surface != NULL) {
    wl_surface_destroy(state->surface);
  }
  if (state->pool != NULL) {
    wl_shm_pool_destroy(state->pool);
  }
  if (state->pool_data != NULL) {
    munmap(state->pool_data, state->pool_size);
  }
  if (state->output_manager != NULL) {
    zxdg_output_manager_v1_destroy(state->output_manager);
  }
  if (state->layer_shell != NULL) {
    zwlr_layer_shell_v1_destroy(state->layer_shell);
  }
  if (state->shm != NULL) {
    wl_shm_destroy(state->shm);
  }
  if (state->compositor != NULL) {
    wl_compositor_destroy(state->compositor);
  }
  if (state->registry != NULL) {
    wl_registry_destroy(state->registry);
  }
  if (state->display != NULL) {
    wl_display_disconnect(state->display);
  }
  free(state);
}

static HudState *create_wayland_state(char *error, size_t error_size) {
  HudState *state = calloc(1, sizeof(*state));
  if (state == NULL) {
    if (error != NULL && error_size > 0) {
      snprintf(error, error_size, "cannot allocate Wayland renderer state");
    }
    return NULL;
  }
  state->display = wl_display_connect(NULL);
  if (state->display == NULL) {
    set_error(state, "cannot connect renderer to the Wayland display");
  }
  if (!state->failed) {
    state->registry = wl_display_get_registry(state->display);
    wl_registry_add_listener(state->registry, &REGISTRY_LISTENER, state);
    if (wl_display_roundtrip(state->display) < 0) {
      set_error(state, "Wayland disconnected while starting renderer");
    }
  }
  if (!state->failed &&
      (state->compositor == NULL || state->shm == NULL ||
       state->layer_shell == NULL || state->output_manager == NULL)) {
    set_error(
        state,
        "Wayland compositor does not provide layer shell and output geometry");
  }
  if (!state->failed) {
    for (size_t index = 0; index < state->output_count; index++) {
      HudOutput *output = &state->outputs[index];
      output->xdg_output = zxdg_output_manager_v1_get_xdg_output(
          state->output_manager, output->output);
      zxdg_output_v1_add_listener(output->xdg_output, &OUTPUT_LISTENER, output);
    }
    if (wl_display_roundtrip(state->display) < 0) {
      set_error(state, "Wayland disconnected while reading output geometry");
    }
  }
  if (state->failed) {
    copy_error(state, error, error_size);
    cleanup(state);
    return NULL;
  }
  return state;
}

__attribute__((visibility("default"))) void *
gazeebo_hud_create(char *error, size_t error_size) {
  HudState *state = create_wayland_state(error, error_size);
  if (state == NULL) {
    return NULL;
  }
  state->frame_width = HUD_WIDTH;
  state->frame_height = HUD_HEIGHT;
  if (!state->failed && create_buffers(state) != 0) {
    state->failed = true;
  }
  if (!state->failed) {
    state->surface = wl_compositor_create_surface(state->compositor);
    state->layer_surface = zwlr_layer_shell_v1_get_layer_surface(
        state->layer_shell, state->surface, NULL,
        ZWLR_LAYER_SHELL_V1_LAYER_OVERLAY, "gazeebo-debug-hud");
    if (state->surface == NULL || state->layer_surface == NULL) {
      set_error(state, "cannot create debug HUD layer surface");
    }
  }
  if (!state->failed) {
    zwlr_layer_surface_v1_add_listener(state->layer_surface, &LAYER_LISTENER,
                                       state);
    zwlr_layer_surface_v1_set_size(state->layer_surface, HUD_WIDTH, HUD_HEIGHT);
    zwlr_layer_surface_v1_set_anchor(state->layer_surface,
                                     ZWLR_LAYER_SURFACE_V1_ANCHOR_TOP |
                                         ZWLR_LAYER_SURFACE_V1_ANCHOR_RIGHT);
    zwlr_layer_surface_v1_set_margin(state->layer_surface, HUD_MARGIN,
                                     HUD_MARGIN, 0, 0);
    zwlr_layer_surface_v1_set_exclusive_zone(state->layer_surface, -1);
    zwlr_layer_surface_v1_set_keyboard_interactivity(
        state->layer_surface,
        ZWLR_LAYER_SURFACE_V1_KEYBOARD_INTERACTIVITY_NONE);
    struct wl_region *empty = wl_compositor_create_region(state->compositor);
    wl_surface_set_input_region(state->surface, empty);
    wl_region_destroy(empty);
    wl_surface_commit(state->surface);
    while (!state->configured && !state->closed && !state->failed) {
      if (wl_display_dispatch(state->display) < 0) {
        set_error(state, "Wayland disconnected while configuring debug HUD");
      }
    }
  }
  if (state->failed || state->closed) {
    copy_error(state, error, error_size);
    cleanup(state);
    return NULL;
  }
  return state;
}

static HudOutput *output_at(HudState *state, double x, double y) {
  for (size_t index = 0; index < state->output_count; index++) {
    HudOutput *output = &state->outputs[index];
    if (output->width > 0 && output->height > 0 && x >= output->x &&
        y >= output->y && x < output->x + output->width &&
        y < output->y + output->height) {
      return output;
    }
  }
  return NULL;
}

static void format_hud_text(HudState *state, const char *region_id, double x,
                            double y, char *text, size_t text_size) {
  HudOutput *output = output_at(state, x, y);
  const char *description = output != NULL && output->description[0] != '\0'
                                ? output->description
                                : "unknown output";
  const char *name =
      output != NULL && output->name[0] != '\0' ? output->name : "unknown";
  snprintf(text, text_size,
           "output: %s\nconnector: %s\nregion: %s\nx: %.0f  y: %.0f",
           description, name, region_id, x, y);
}

__attribute__((visibility("default"))) int
gazeebo_hud_update(void *handle, const char *region_id, double x, double y,
                   char *error, size_t error_size) {
  HudState *state = handle;
  if (state == NULL || region_id == NULL || state->closed) {
    if (error != NULL && error_size > 0) {
      snprintf(error, error_size, "debug HUD is closed");
    }
    return -1;
  }
  (void)wl_display_dispatch_pending(state->display);
  HudBuffer *frame = available_buffer(state);
  if (frame == NULL) {
    if (wl_display_roundtrip(state->display) < 0) {
      set_error(state, "Wayland disconnected while updating debug HUD");
    }
    frame = available_buffer(state);
  }
  if (frame == NULL) {
    set_error(state, "debug HUD has no available drawing buffer");
  }
  if (!state->failed) {
    char text[1024];
    format_hud_text(state, region_id, x, y, text, sizeof(text));
    render_hud(state, frame, text);
    frame->busy = true;
    wl_surface_attach(state->surface, frame->buffer, 0, 0);
    wl_surface_damage_buffer(state->surface, 0, 0, state->frame_width,
                             state->frame_height);
    wl_surface_commit(state->surface);
    if (wl_display_flush(state->display) < 0 && errno != EAGAIN) {
      set_error(state, "cannot flush debug HUD update");
    }
  }
  if (state->failed) {
    copy_error(state, error, error_size);
    return -1;
  }
  return 0;
}

__attribute__((visibility("default"))) void gazeebo_hud_destroy(void *handle) {
  cleanup(handle);
}

__attribute__((visibility("default"))) void *
gazeebo_game_create(int32_t region_x, int32_t region_y, int32_t region_width,
                    int32_t region_height, char *error, size_t error_size) {
  if (region_width <= 0 || region_height <= 0) {
    if (error != NULL && error_size > 0) {
      snprintf(error, error_size, "calibration game region is invalid");
    }
    return NULL;
  }
  HudState *state = create_wayland_state(error, error_size);
  if (state == NULL) {
    return NULL;
  }
  HudOutput *output = output_at(state, region_x + region_width / 2.0,
                                region_y + region_height / 2.0);
  if (output == NULL) {
    set_error(state, "selected portal region does not match a Wayland output");
  }
  state->frame_width = region_width;
  state->frame_height = region_height;
  if (!state->failed && create_buffers(state) != 0) {
    state->failed = true;
  }
  if (!state->failed) {
    state->surface = wl_compositor_create_surface(state->compositor);
    state->layer_surface = zwlr_layer_shell_v1_get_layer_surface(
        state->layer_shell, state->surface, output->output,
        ZWLR_LAYER_SHELL_V1_LAYER_OVERLAY, GAME_NAMESPACE);
    if (state->surface == NULL || state->layer_surface == NULL) {
      set_error(state, "cannot create calibration game layer surface");
    }
  }
  if (!state->failed) {
    zwlr_layer_surface_v1_add_listener(state->layer_surface, &LAYER_LISTENER,
                                       state);
    zwlr_layer_surface_v1_set_size(state->layer_surface, region_width,
                                   region_height);
    zwlr_layer_surface_v1_set_anchor(state->layer_surface,
                                     ZWLR_LAYER_SURFACE_V1_ANCHOR_TOP |
                                         ZWLR_LAYER_SURFACE_V1_ANCHOR_RIGHT |
                                         ZWLR_LAYER_SURFACE_V1_ANCHOR_BOTTOM |
                                         ZWLR_LAYER_SURFACE_V1_ANCHOR_LEFT);
    zwlr_layer_surface_v1_set_exclusive_zone(state->layer_surface, -1);
    zwlr_layer_surface_v1_set_keyboard_interactivity(
        state->layer_surface,
        ZWLR_LAYER_SURFACE_V1_KEYBOARD_INTERACTIVITY_NONE);
    struct wl_region *empty = wl_compositor_create_region(state->compositor);
    wl_surface_set_input_region(state->surface, empty);
    wl_region_destroy(empty);
    wl_surface_commit(state->surface);
    while (!state->configured && !state->closed && !state->failed) {
      if (wl_display_dispatch(state->display) < 0) {
        set_error(state,
                  "Wayland disconnected while configuring calibration game");
      }
    }
  }
  if (state->failed || state->closed) {
    copy_error(state, error, error_size);
    cleanup(state);
    return NULL;
  }
  return state;
}

__attribute__((visibility("default"))) int
gazeebo_game_show_target(void *handle, double x, double y, double diameter,
                         const char *label, char *error, size_t error_size) {
  HudState *state = handle;
  if (state == NULL || label == NULL || state->closed) {
    if (error != NULL && error_size > 0) {
      snprintf(error, error_size, "calibration game is closed");
    }
    return -1;
  }
  if (diameter <= 0.0 || x - diameter / 2.0 < 0.0 || y - diameter / 2.0 < 0.0 ||
      x + diameter / 2.0 > state->frame_width ||
      y + diameter / 2.0 > state->frame_height) {
    if (error != NULL && error_size > 0) {
      snprintf(error, error_size, "calibration target is outside its display");
    }
    return -1;
  }
  (void)wl_display_dispatch_pending(state->display);
  HudBuffer *frame = available_buffer(state);
  if (frame == NULL) {
    if (wl_display_roundtrip(state->display) < 0) {
      set_error(state, "Wayland disconnected while updating calibration game");
    }
    frame = available_buffer(state);
  }
  if (frame == NULL) {
    set_error(state, "calibration game has no available drawing buffer");
  }
  if (!state->failed) {
    render_game(state, frame, x, y, diameter, label);
    frame->busy = true;
    wl_surface_attach(state->surface, frame->buffer, 0, 0);
    wl_surface_damage_buffer(state->surface, 0, 0, state->frame_width,
                             state->frame_height);
    wl_surface_commit(state->surface);
    if (wl_display_flush(state->display) < 0 && errno != EAGAIN) {
      set_error(state, "cannot flush calibration game update");
    }
  }
  if (state->failed) {
    copy_error(state, error, error_size);
    return -1;
  }
  return 0;
}

__attribute__((visibility("default"))) int
gazeebo_game_hide(void *handle, char *error, size_t error_size) {
  HudState *state = handle;
  if (state == NULL || state->closed) {
    if (error != NULL && error_size > 0) {
      snprintf(error, error_size, "calibration game is closed");
    }
    return -1;
  }
  (void)wl_display_dispatch_pending(state->display);
  HudBuffer *frame = available_buffer(state);
  if (frame == NULL && wl_display_roundtrip(state->display) >= 0) {
    frame = available_buffer(state);
  }
  if (frame == NULL) {
    set_error(state, "calibration game has no available clearing buffer");
  }
  if (!state->failed) {
    memset(frame->pixels, 0,
           (size_t)state->frame_width * (size_t)state->frame_height *
               sizeof(*frame->pixels));
    frame->busy = true;
    wl_surface_attach(state->surface, frame->buffer, 0, 0);
    wl_surface_damage_buffer(state->surface, 0, 0, state->frame_width,
                             state->frame_height);
    wl_surface_commit(state->surface);
    if (wl_display_flush(state->display) < 0 && errno != EAGAIN) {
      set_error(state, "cannot flush calibration game clear");
    }
  }
  if (state->failed) {
    copy_error(state, error, error_size);
    return -1;
  }
  return 0;
}

__attribute__((visibility("default"))) void gazeebo_game_destroy(void *handle) {
  cleanup(handle);
}
