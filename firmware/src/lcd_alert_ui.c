#include "lcd_alert_ui.h"

#include <rtthread.h>
#include <rtdevice.h>
#include <stdint.h>
#include <string.h>

#define UI_WIDTH          480
#define UI_HEIGHT         800

#define RGB565(r, g, b) ((uint16_t)((((r) & 0xf8) << 8) | (((g) & 0xfc) << 3) | ((b) >> 3)))

#define COLOR_BG       RGB565(7, 18, 30)
#define COLOR_PANEL    RGB565(14, 34, 50)
#define COLOR_TEXT     RGB565(238, 246, 252)
#define COLOR_MUTED    RGB565(135, 160, 176)
#define COLOR_OK       RGB565(42, 205, 145)
#define COLOR_VERIFY   RGB565(255, 185, 55)
#define COLOR_ALERT    RGB565(244, 67, 54)
#define COLOR_BAR_BG   RGB565(35, 57, 72)

static rt_device_t ui_lcd;
static uint16_t *ui_fb;

typedef enum
{
    UI_SCREEN_NONE = 0,
    UI_SCREEN_STARTING,
    UI_SCREEN_MONITORING,
    UI_SCREEN_CANDIDATE,
    UI_SCREEN_ALERT,
    UI_SCREEN_ERROR
} ui_screen_t;

static ui_screen_t current_screen = UI_SCREEN_NONE;
static int current_class = -1;
static int current_hit_count = -1;
static int current_required_hits = -1;

static const char *const class_lines[6][2] =
{
    {"BABY", "CRYING"},
    {"GLASS", "BREAKING"},
    {"ALARM", "SIREN"},
    {"DOOR", "KNOCK"},
    {"DOORBELL", ""},
    {"BACKGROUND", ""}
};

/* Compact 5x7 uppercase bitmap font; each byte is one vertical column. */
static const uint8_t font_letters[26][5] =
{
    {0x7e,0x11,0x11,0x11,0x7e}, {0x7f,0x49,0x49,0x49,0x36},
    {0x3e,0x41,0x41,0x41,0x22}, {0x7f,0x41,0x41,0x22,0x1c},
    {0x7f,0x49,0x49,0x49,0x41}, {0x7f,0x09,0x09,0x09,0x01},
    {0x3e,0x41,0x49,0x49,0x7a}, {0x7f,0x08,0x08,0x08,0x7f},
    {0x00,0x41,0x7f,0x41,0x00}, {0x20,0x40,0x41,0x3f,0x01},
    {0x7f,0x08,0x14,0x22,0x41}, {0x7f,0x40,0x40,0x40,0x40},
    {0x7f,0x02,0x0c,0x02,0x7f}, {0x7f,0x04,0x08,0x10,0x7f},
    {0x3e,0x41,0x41,0x41,0x3e}, {0x7f,0x09,0x09,0x09,0x06},
    {0x3e,0x41,0x51,0x21,0x5e}, {0x7f,0x09,0x19,0x29,0x46},
    {0x46,0x49,0x49,0x49,0x31}, {0x01,0x01,0x7f,0x01,0x01},
    {0x3f,0x40,0x40,0x40,0x3f}, {0x1f,0x20,0x40,0x20,0x1f},
    {0x3f,0x40,0x38,0x40,0x3f}, {0x63,0x14,0x08,0x14,0x63},
    {0x07,0x08,0x70,0x08,0x07}, {0x61,0x51,0x49,0x45,0x43}
};

static const uint8_t font_digits[10][5] =
{
    {0x3e,0x51,0x49,0x45,0x3e}, {0x00,0x42,0x7f,0x40,0x00},
    {0x42,0x61,0x51,0x49,0x46}, {0x21,0x41,0x45,0x4b,0x31},
    {0x18,0x14,0x12,0x7f,0x10}, {0x27,0x45,0x45,0x45,0x39},
    {0x3c,0x4a,0x49,0x49,0x30}, {0x01,0x71,0x09,0x05,0x03},
    {0x36,0x49,0x49,0x49,0x36}, {0x06,0x49,0x49,0x29,0x1e}
};

static const uint8_t *glyph_for(char ch)
{
    static const uint8_t blank[5]   = {0,0,0,0,0};
    static const uint8_t percent[5] = {0x63,0x13,0x08,0x64,0x63};
    static const uint8_t slash[5]   = {0x20,0x10,0x08,0x04,0x02};
    static const uint8_t colon[5]   = {0x00,0x36,0x36,0x00,0x00};
    static const uint8_t dash[5]    = {0x08,0x08,0x08,0x08,0x08};
    static const uint8_t bang[5]    = {0x00,0x00,0x5f,0x00,0x00};

    if ((ch >= 'A') && (ch <= 'Z')) return font_letters[ch - 'A'];
    if ((ch >= '0') && (ch <= '9')) return font_digits[ch - '0'];
    if (ch == '%') return percent;
    if (ch == '/') return slash;
    if (ch == ':') return colon;
    if (ch == '-') return dash;
    if (ch == '!') return bang;
    return blank;
}

static void fill_rect(int x, int y, int w, int h, uint16_t color)
{
    if ((ui_fb == RT_NULL) || (w <= 0) || (h <= 0)) return;
    if (x < 0) { w += x; x = 0; }
    if (y < 0) { h += y; y = 0; }
    if ((x + w) > UI_WIDTH) w = UI_WIDTH - x;
    if ((y + h) > UI_HEIGHT) h = UI_HEIGHT - y;

    for (int row = y; row < (y + h); row++)
    {
        uint16_t *dst = ui_fb + (row * UI_WIDTH) + x;
        for (int col = 0; col < w; col++) dst[col] = color;
    }
}

static int text_width(const char *text, int scale)
{
    return (int)strlen(text) * 6 * scale - scale;
}

static void draw_text(int x, int y, const char *text, int scale, uint16_t color)
{
    while (*text != '\0')
    {
        const uint8_t *glyph = glyph_for(*text++);
        for (int col = 0; col < 5; col++)
        {
            for (int row = 0; row < 7; row++)
            {
                if ((glyph[col] >> row) & 1U)
                {
                    fill_rect(x + col * scale, y + row * scale, scale, scale, color);
                }
            }
        }
        x += 6 * scale;
    }
}

static void draw_centered(int y, const char *text, int scale, uint16_t color)
{
    draw_text((UI_WIDTH - text_width(text, scale)) / 2, y, text, scale, color);
}

static void draw_frame(uint16_t accent, const char *status)
{
    fill_rect(0, 0, UI_WIDTH, UI_HEIGHT, COLOR_BG);
    draw_centered(38, "SOUND MONITOR", 4, COLOR_TEXT);
    fill_rect(44, 104, 392, 54, accent);
    draw_centered(117, status, 4, COLOR_BG);
    fill_rect(24, 184, 432, 342, COLOR_PANEL);
    fill_rect(24, 550, 432, 172, COLOR_PANEL);
    draw_centered(746, "AI ACOUSTIC DETECTION", 2, COLOR_MUTED);
}

static void draw_class_name(int class_index, uint16_t color)
{
    if ((class_index < 0) || (class_index >= 6)) class_index = 5;
    if (class_lines[class_index][1][0] == '\0')
    {
        draw_centered(322, class_lines[class_index][0], 5, color);
    }
    else
    {
        draw_centered(274, class_lines[class_index][0], 6, color);
        draw_centered(354, class_lines[class_index][1], 6, color);
    }
}

static void draw_alert_message(int class_index)
{
    if ((class_index < 0) || (class_index >= 5)) class_index = 5;

    draw_centered(224, "ALERT!", 7, COLOR_ALERT);

    if (class_lines[class_index][1][0] == '\0')
    {
        draw_centered(356, class_lines[class_index][0], 5, COLOR_TEXT);
    }
    else
    {
        draw_centered(324, class_lines[class_index][0], 5, COLOR_TEXT);
        draw_centered(398, class_lines[class_index][1], 5, COLOR_TEXT);
    }
}

static void draw_confidence(int confidence_percent, uint16_t accent)
{
    char percent[8];
    if (confidence_percent < 0) confidence_percent = 0;
    if (confidence_percent > 100) confidence_percent = 100;
    rt_snprintf(percent, sizeof(percent), "%d%%", confidence_percent);

    draw_centered(578, "CONFIDENCE", 3, COLOR_MUTED);
    draw_centered(621, percent, 5, COLOR_TEXT);
    fill_rect(54, 686, 372, 14, COLOR_BAR_BG);
    fill_rect(54, 686, (372 * confidence_percent) / 100, 14, accent);
}

static void present(void)
{
    struct rt_device_rect_info rect = {0, 0, UI_WIDTH, UI_HEIGHT};
    rt_device_control(ui_lcd, RTGRAPHIC_CTRL_RECT_UPDATE, &rect);
}

int lcd_alert_ui_init(void)
{
    struct rt_device_graphic_info info;
    ui_lcd = rt_device_find("lcd");
    if (ui_lcd == RT_NULL) return -RT_ENOSYS;
    if (rt_device_control(ui_lcd, RTGRAPHIC_CTRL_GET_INFO, &info) != RT_EOK) return -RT_ERROR;
    if ((info.framebuffer == RT_NULL) || (info.width != UI_WIDTH) || (info.height != UI_HEIGHT)) return -RT_EINVAL;

    ui_fb = (uint16_t *)info.framebuffer;
    draw_frame(COLOR_OK, "STARTING");
    draw_centered(322, "INITIALIZING", 4, COLOR_TEXT);
    present();
    current_screen = UI_SCREEN_STARTING;
    return RT_EOK;
}

void lcd_alert_ui_show_monitoring(void)
{
    if (ui_fb == RT_NULL) return;

    draw_frame(COLOR_OK, "LISTENING");
    draw_class_name(5, COLOR_TEXT);
    draw_centered(472, "NO ALERT", 3, COLOR_OK);
    draw_centered(624, "MONITORING", 4, COLOR_TEXT);
    present();
    current_screen = UI_SCREEN_MONITORING;
    current_class = 5;
}

void lcd_alert_ui_show_candidate(int class_index, int confidence_percent,
                                 int hit_count, int required_hits)
{
    char progress[20];
    if (ui_fb == RT_NULL) return;

    draw_frame(COLOR_VERIFY, "VERIFYING");
    draw_class_name(class_index, COLOR_TEXT);
    rt_snprintf(progress, sizeof(progress), "CHECK %d/%d", hit_count, required_hits);
    draw_centered(472, progress, 3, COLOR_VERIFY);
    draw_confidence(confidence_percent, COLOR_VERIFY);
    present();
    current_screen = UI_SCREEN_CANDIDATE;
    current_class = class_index;
    current_hit_count = hit_count;
    current_required_hits = required_hits;
}

void lcd_alert_ui_show_alert(int class_index, int confidence_percent)
{
    if (ui_fb == RT_NULL) return;

    draw_frame(COLOR_ALERT, "ALERT!");
    draw_alert_message(class_index);
    draw_centered(480, "SOUND DETECTED", 3, COLOR_ALERT);
    draw_confidence(confidence_percent, COLOR_ALERT);
    present();
    current_screen = UI_SCREEN_ALERT;
    current_class = class_index;
}

void lcd_alert_ui_show_error(const char *message)
{
    if (ui_fb == RT_NULL) return;
    if (current_screen == UI_SCREEN_ERROR) return;

    draw_frame(COLOR_ALERT, "SYSTEM ERROR");
    draw_centered(314, message, 3, COLOR_TEXT);
    draw_centered(390, "CHECK SERIAL LOG", 3, COLOR_MUTED);
    present();
    current_screen = UI_SCREEN_ERROR;
}

const char *lcd_alert_ui_state_name(void)
{
    switch (current_screen)
    {
        case UI_SCREEN_STARTING:   return "STARTING";
        case UI_SCREEN_MONITORING: return "MONITORING";
        case UI_SCREEN_CANDIDATE:  return "VERIFYING";
        case UI_SCREEN_ALERT:      return "ALERT";
        case UI_SCREEN_ERROR:      return "ERROR";
        default:                   return "NOT READY";
    }
}

static int lcd_ui_test(void)
{
    if (ui_fb == RT_NULL)
    {
        rt_kprintf("LCD UI is not initialized.\n");
        return -RT_ERROR;
    }

    current_screen = UI_SCREEN_NONE;
    rt_kprintf("LCD UI test: monitoring -> verifying -> alert.\n");
    lcd_alert_ui_show_monitoring();
    rt_thread_mdelay(1500);
    lcd_alert_ui_show_candidate(4, 82, 1, 2);
    rt_thread_mdelay(1500);
    lcd_alert_ui_show_alert(4, 91);
    return RT_EOK;
}
MSH_CMD_EXPORT(lcd_ui_test, preview monitoring verifying and alert LCD screens);
