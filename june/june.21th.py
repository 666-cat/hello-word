import math
import numpy as np
source=[{
	"size": ('height', 'width'),
	"box": {
		1: [
			[-1000, 1000],
			[1000, -1000], math.pi / 4
		]},
        "lines": {
		1: [
			[[100, 100], [-100, 100]],
			[[100, -100], [-100, -100]],
		]}
        }]
perfect_liners={}#完美线段
perfectlines=[]
rotate_box={}#旋转后的四个点
perfect_points={}
perfectpoints=[]
def rotate_box_corners(point1, point2, rot):#输出旋转后四个点的坐标
    # 计算方框中心点
    center_x = (point1[0] + point2[0]) / 2
    center_y = (point1[1] + point2[1]) / 2
    x1, y1 = point1
    x2, y2 = point2
    corners = [(x1, y1), ((x1+x2+y1-y2)/2, (x2-x1+y1+y2)/2), (x2, y2), ((x1+x2-y1+y2)/2, (x1-x2+y1+y2)/2)]
    rotated_corners = []
    for x, y in corners:
        dx = x - center_x
        dy = y - center_y
        # 应用旋转公式（顺时针）
        new_dx = dx * math.cos(rot) + dy * math.sin(rot)
        new_dy = -dx * math.sin(rot) + dy * math.cos(rot)
         # 转换回绝对坐标
        new_x = new_dx + center_x
        new_y = new_dy + center_y
        rotated_corners.append([new_x, new_y])
    return rotated_corners
def lines_in_box_rect(point1, point2, corners):#输出完美线段的两个点
    length=math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)
    ab=np.array(corners[0])-np.array(corners[1])
    bc=np.array(corners[1])-np.array(corners[2])
    cd=np.array(corners[2])-np.array(corners[3])
    da=np.array(corners[3])-np.array(corners[0])
    am=np.array(point1)-np.array(corners[0])
    bm=np.array(point1)-np.array(corners[1])
    cm=np.array(point1)-np.array(corners[2])
    dm=np.array(point1)-np.array(corners[3])
    am2=np.array(point2)-np.array(corners[0])
    bm2=np.array(point2)-np.array(corners[1])
    cm2=np.array(point2)-np.array(corners[2])
    dm2=np.array(point2)-np.array(corners[3])
    p_ab=np.dot(ab,am)
    p_bc=np.dot(bc,bm)
    p_cd=np.dot(cd,cm)
    p_da=np.dot(da,dm)
    p_ab2=np.dot(ab,am2)
    p_bc2=np.dot(bc,bm2)
    p_cd2=np.dot(cd,cm2)
    p_da2=np.dot(da,dm2)
    #print(p_ab,p_bc,p_cd,p_da,p_ab2,p_bc2,p_cd2,p_da2,length)
    if p_ab<0 and p_bc<0 and p_cd<0 and p_da<0 and p_ab2<0 and p_bc2<0 and p_cd2<0 and p_da2<0 and length>30:
        return point1,point2
    else:
        return None
def perfect_point(point1,point2,corners):#输出完美点的坐标
    y2=point2[1]
    y1=point1[1]
    x2=point2[0]
    x1=point1[0]
    if x1!=x2:
        a1=(y2-y1)
        b1=(x1-x2)
        c1=(x2*y1-x1*y2)
    else:
        a1=1
        b1=0
        c1=-x1
    edges = []
    for i in range(4):
        edges.append((corners[i], corners[(i+1)%4]))
    for edge in edges:
        (x3, y3), (x4, y4) = edge
        if x3!=x4:
            a2=(y4-y3)
            b2=(x3-x4)
            c2=(x4*y3-x3*y4)
        else:
            a2=1
            b2=0
            c2=-x3
        if a1*b2==a2*b1:
            print('平行')
        else:
            x=(c2*b1-c1*b2)/(a1*b2-a2*b1)
            y=(c1*a2-c2*a1)/(a1*b2-a2*b1)
            if x>=min(x3,x4) and x<=max(x3,x4) and y>=min(y3,y4) and y<=max(y3,y4):
                perfectpoints.append([x,y])
for box,lines,i in zip(source[0]['box'].values(),source[0]['lines'].values(),range(len(source[0]['box']))):
    i+=1
    result = rotate_box_corners(box[0], box[1], box[2])
    rotate_box[str(i)]=(result)
    print('旋转后坐标')
    print(rotate_box)
    for line in lines:
        '''print(line[0])
        print(line[1])
        print(type(line[0]))
        print(rotate_box[str(i)])
        print(type(rotate_box[str(i)]))'''
        perfect_liner=lines_in_box_rect(line[0],line[1],rotate_box[str(i)])
        if perfect_liner!=None:
            perfectlines.append(perfect_liner)
        else:
            print('第'+str(i)+'个矩形的第'+str(lines.index(line)+1)+'条线段不是完美线段')
    perfect_liners[str(i)]=perfectlines
    print("完美线段")
    print(perfect_liners)
    for perfect_liner in perfect_liners.values():
        #print(perfect_liner)
        #print(rotate_box[str(i)])
        for j in range(len(perfect_liner)):
            perfect_point(perfect_liner[j][0],perfect_liner[j][1],rotate_box[str(i)])
            perfect_points[str(i)]=perfectpoints
        print("完美线段延长线的交点")
        print(perfect_points)