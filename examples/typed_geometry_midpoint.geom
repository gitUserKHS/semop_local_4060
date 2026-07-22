# Typed geometry DSL: a midpoint creates equal symbolic segment lengths.
point A, B, M

let AM: Segment = segment(A, M)

assume midpoint(M, A, B)
prove collinear(A, M, B)
prove equal_length(AM, segment(M, B))
