from s02_parse_vtt_file import *
assert(((r"""00:00:00 [Music]
00:00:00 welcome to BASF we create chemistry so
00:00:05 it makes sense that we should
00:00:06 familiarize you with the basic chemistry
00:00:08 taught in our poly urethanes Academy
00:00:11 we're going to simplify things a bit in
00:00:13 this video and at the same time cover a
00:00:16 lot of topics so let's get started first
00:00:18 let's introduce you to two of our
00:00:21 leading characters by societies or ISO
00:00:25 and resin let's talk about ISO first
00:00:29 when we make ISO we do so in very large
00:00:32 quantities for our purposes today there
00:00:35 are only a few types of eye soaps pure
00:00:38 MD eyes and TV eyes that's their
00:00:40 nicknames form long and squiggly
00:00:43 chemical structures that's because they
00:00:45 have fewer places to connect to they are
00:00:47 generally used to make flexible products
00:00:50 like seat cushions mattresses and
00:00:52 sealants polymeric MD eyes have many
00:00:55 more places to plug into which creates
00:00:58 more of the structure they are generally
00:01:00 used to make you guessed it rigid
00:01:03 products like picnic coolers foam
00:01:05 insulation and wood boards now when our
00:01:09 customers make a resin they create a
00:01:11 custom formula of additives that include
00:01:14 polygons also supplied by BASF which are
00:01:18 the backbone of the mix polyols make the
00:01:21 majority of the mix kind of like flour
00:01:24 is to a cake batter
00:01:25 polyols determine the physical
00:01:27 properties of the product like how soft
00:01:30 or hard the product is
00:01:32 catalysts they control the speed of the
00:01:35 chemical reaction and how quickly it
00:01:37 cures surfactants determine the cell
00:01:40 structure and influence the flow
00:01:42 pigments determine the color flame
00:01:45 retardants make it savory adhesion
00:01:48 promoters make it stickier and finally
00:01:50 blowing agents help determine the
00:01:53 density and foaming action
00:01:55 at BASF we're proud to supply raw
00:01:58 materials that help our customers
00:02:00 innovate and succeed on ISOs and polyols
00:02:04 combined to make custom formulas for our
00:02:06 customers custom formulas that produce
00:02:09 unique products that are flexible rigid
00:02:33 just the way end-users like them so
00:02:33 there you have it the basics of
00:02:36 polyurethanes from BASF we create
""")==(parse_vtt_file("cW3tzRzTHKI.en.vtt"))))