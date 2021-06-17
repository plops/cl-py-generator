// File: mem.v
// Generated by MyHDL 0.11
// Date: Thu Jun 17 08:28:41 2021


`timescale 1ns/10ps

module mem (
    clk,
    adr,
    we,
    di,
    do
);


input clk;
input [15:0] adr;
input we;
input [7:0] di;
output [7:0] do;
reg [7:0] do;

reg [7:0] ram [0:256-1];



always @(posedge clk) begin: MEM_LOGIC
    if (we) begin
        ram[adr] <= di;
    end
    else begin
        if ((adr < 13)) begin
            do <= 0;
        end
        else begin
            do <= ram[adr];
        end
    end
end

endmodule
