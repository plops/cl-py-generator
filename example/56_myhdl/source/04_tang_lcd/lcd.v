// File: lcd.v
// Generated by MyHDL 0.11
// Date: Fri Jun 18 18:34:20 2021


`timescale 1ns/10ps

module lcd (
    pixel_clk,
    n_rst,
    lcd_de,
    lcd_hsync,
    lcd_vsync,
    lcd_r,
    lcd_g,
    lcd_b
);


input pixel_clk;
input n_rst;
output lcd_de;
reg lcd_de;
output lcd_hsync;
reg lcd_hsync;
output lcd_vsync;
reg lcd_vsync;
output [4:0] lcd_r;
reg [4:0] lcd_r;
output [5:0] lcd_g;
reg [5:0] lcd_g;
output [4:0] lcd_b;
reg [4:0] lcd_b;

reg [9:0] pixel_count;
reg [9:0] line_count;
reg [7:0] frame_count;
reg frame_odd;



always @(posedge pixel_clk, negedge n_rst) begin: LCD_LOGIC_COUNT
    if ((n_rst == 0)) begin
        line_count <= 0;
        pixel_count <= 0;
    end
    else begin
        if ((pixel_count == 505)) begin
            line_count <= (line_count + 1);
            pixel_count <= 0;
        end
        else begin
            if ((line_count == 323)) begin
                line_count <= 0;
                pixel_count <= 0;
            end
            else begin
                pixel_count <= (pixel_count + 1);
            end
        end
    end
end


always @(pixel_count, line_count) begin: LCD_LOGIC_SYNC
    if (((1 <= pixel_count) & (pixel_count <= (480 + 5)))) begin
        lcd_hsync = 0;
    end
    else begin
        lcd_hsync = 1;
    end
    if (((5 <= line_count) & (line_count <= 323))) begin
        lcd_vsync = 0;
    end
    else begin
        lcd_vsync = 1;
    end
    if ((((5 <= pixel_count) & (pixel_count <= (480 + 5))) & ((6 <= line_count) & (line_count <= (272 + 5))))) begin
        lcd_de = 1;
    end
    else begin
        lcd_de = 0;
    end
end


always @(frame_count, frame_odd, pixel_count) begin: LCD_LOGIC_PATTERN
    if (((frame_count + pixel_count) < 200)) begin
        lcd_r = 0;
    end
    else if (((frame_count + pixel_count) < 240)) begin
        lcd_r = 1;
    end
    else if (((frame_count + pixel_count) < 280)) begin
        lcd_r = 2;
    end
    else if (((frame_count + pixel_count) < 320)) begin
        lcd_r = 4;
    end
    else if (((frame_count + pixel_count) < 360)) begin
        lcd_r = 8;
    end
    else if (((frame_count + pixel_count) < 480)) begin
        lcd_r = 16;
    end
    else begin
        lcd_r = 0;
        lcd_b = 0;
    end
    if ((frame_odd & (pixel_count < 100))) begin
        lcd_g = 0;
    end
    else if ((frame_odd & (pixel_count < 140))) begin
        lcd_g = 1;
    end
    else if ((frame_odd & (pixel_count < 180))) begin
        lcd_g = 2;
    end
    else if ((frame_odd & (pixel_count < 220))) begin
        lcd_g = 4;
    end
    else if ((frame_odd & (pixel_count < 260))) begin
        lcd_g = 8;
    end
    else if ((frame_odd & (pixel_count < 380))) begin
        lcd_g = 16;
    end
    else begin
        lcd_g = 0;
    end
end


always @(posedge lcd_vsync) begin: LCD_LOGIC_FRAME
    frame_count <= (frame_count + 1);
    frame_odd <= (!frame_odd);
end

endmodule
