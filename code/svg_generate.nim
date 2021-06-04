import nimsvg

buildSvgFile("figures/nim/diagram-1.svg"):
  let size = 100
  svg(width=size, height=size):
    for i in 0 .. 20:
      for j in 0 .. 20:
        let x = i*5
        let y = j*5 
        circle(cx=x, cy=y, r=2, stroke="#111122", fill="#E0E0F0", `fill-opacity`=0.5)

