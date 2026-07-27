#ifndef LCD_ALERT_UI_H
#define LCD_ALERT_UI_H

int lcd_alert_ui_init(void);
void lcd_alert_ui_show_monitoring(void);
void lcd_alert_ui_show_candidate(int class_index, int confidence_percent,
                                 int hit_count, int required_hits);
void lcd_alert_ui_show_alert(int class_index, int confidence_percent);
void lcd_alert_ui_show_error(const char *message);
const char *lcd_alert_ui_state_name(void);

#endif
