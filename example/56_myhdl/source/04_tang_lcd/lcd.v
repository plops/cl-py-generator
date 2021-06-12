// File: lcd.v
// Generated by MyHDL 0.11
// Date: Sat Jun 12 08:21:21 2021


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
input lcd_de;
input lcd_hsync;
input lcd_vsync;
output [4:0] lcd_r;
reg [4:0] lcd_r;
output [4:0] lcd_g;
reg [4:0] lcd_g;
output [5:0] lcd_b;
reg [5:0] lcd_b;

reg [10:0] pixel_count;
reg [9:0] line_count;
reg [9:0] data_r;
reg [9:0] data_g;
reg [9:0] data_b;



always @(posedge pixel_clk, negedge n_rst) begin: LCD_LOGIC_COUNT
    if ((n_rst == 0)) begin
        line_count <= 0;
        pixel_count <= 0;
    end
    else begin
        if ((pixel_count == 1192)) begin
            line_count <= (line_count + 1);
            pixel_count <= 0;
        end
        else begin
            if ((line_count == 548)) begin
                line_count <= 0;
                pixel_count <= 0;
            end
            else begin
                pixel_count <= (pixel_count + 1);
            end
        end
    end
end


always @(posedge pixel_clk, negedge n_rst) begin: LCD_LOGIC_DATA
    if ((n_rst == 0)) begin
        data_r <= 0;
        data_b <= 0;
        data_g <= 0;
    end
end


always @(line_count, pixel_count) begin: LCD_LOGIC_SYNC
    reg lcd_hsync;
    reg lcd_vsync;
    reg lcd_de;
    if (((1 <= pixel_count) & (pixel_count <= (800 + 182)))) begin
        lcd_hsync = 0;
    end
    else begin
        lcd_hsync = 1;
    end
    if (((5 <= line_count) & (line_count <= 548))) begin
        lcd_vsync = 0;
    end
    else begin
        lcd_vsync = 1;
    end
    if ((((182 <= pixel_count) & (pixel_count <= (800 + 182))) & ((6 <= line_count) & (line_count <= (480 + 5))))) begin
        lcd_de = 1;
    end
    else begin
        lcd_de = 0;
    end
end


always @(pixel_count) begin: LCD_LOGIC_PATTERN
    if ((pixel_count < 200)) begin
        lcd_r = 0;
    end
    else begin
        if ((pixel_count < 240)) begin
            lcd_r = 1;
        end
        else begin
            if ((pixel_count < 280)) begin
                lcd_r = 2;
            end
            else begin
                if ((pixel_count < 320)) begin
                    lcd_r = 4;
                end
                else begin
                    if ((pixel_count < 360)) begin
                        lcd_r = 8;
                    end
                    else begin
                        if ((pixel_count < 400)) begin
                            lcd_r = 16;
                        end
                        else begin
                            lcd_r = 0;
                        end
                    end
                end
            end
        end
    end
    if ((pixel_count < 440)) begin
        lcd_b = 0;
    end
    else begin
        if ((pixel_count < 480)) begin
            lcd_b = 1;
        end
        else begin
            if ((pixel_count < 520)) begin
                lcd_b = 2;
            end
            else begin
                if ((pixel_count < 560)) begin
                    lcd_b = 4;
                end
                else begin
                    if ((pixel_count < 600)) begin
                        lcd_b = 8;
                    end
                    else begin
                        if ((pixel_count < 640)) begin
                            lcd_b = 16;
                        end
                        else begin
                            lcd_b = 0;
                        end
                    end
                end
            end
        end
    end
    if ((pixel_count < 640)) begin
        lcd_g = 0;
    end
    else begin
        if ((pixel_count < 680)) begin
            lcd_g = 1;
        end
        else begin
            if ((pixel_count < 720)) begin
                lcd_g = 2;
            end
            else begin
                if ((pixel_count < 760)) begin
                    lcd_g = 4;
                end
                else begin
                    if ((pixel_count < 800)) begin
                        lcd_g = 8;
                    end
                    else begin
                        if ((pixel_count < 840)) begin
                            lcd_g = 16;
                        end
                        else begin
                            lcd_g = 0;
                        end
                    end
                end
            end
        end
    end
end

endmodule
