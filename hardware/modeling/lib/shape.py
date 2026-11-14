"""모양 다듬기 (manifold3d)."""
import manifold3d as m3


def fillet(A, r, seg=12):
    """바깥(볼록) 모서리를 반지름 r 로 둥글게: 구로 깎았다가(침식) 다시 불림(팽창) = 열림 연산.
    재료가 늘지 않아 짝부품과 새로 부딪히지 않는다. 두께 2r 보다 얇은 곳은 사라지므로 r 은 가장 얇은 벽의 절반 아래로.
    나사산은 이 뒤에 뚫는다(산 끝이 둥글어지지 않게). 계산 찌꺼기(부피 0 조각)는 버린다."""
    s = m3.Manifold.sphere(r, seg)
    B = A.minkowski_difference(s).minkowski_sum(s)
    keep = [p for p in B.decompose() if abs(p.volume()) > 0.01]
    out = m3.Manifold()
    for p in keep:
        out += p
    return out.simplify(0.001)          # 구 면끼리 만나며 생긴 넓이 0 삼각형 정리
