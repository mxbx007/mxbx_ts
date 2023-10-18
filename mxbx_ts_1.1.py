# 新增StressShow命令

import numpy as np
import pyautogui
import copy
import cv2
from sklearn import cluster


class TuSe:
    def __init__(self):
        print('欢迎使用')

    def GetCapture(self, stax, stay, endx, endy):
        w = endx - stax
        h = endy - stay
        im = pyautogui.screenshot(region=(stax, stay, w, h))
        # im = cv2.cvtColor(np.array(im), cv2.COLOR_BGR2RGB)
        return np.array(im)

    def FindPic(self, x1, y1, x2, y2, path, thd):
        '''
        找图
        :param x1: 起点X
        :param y1: 起点Y
        :param x2: 终点X
        :param y2: 终点Y
        :param path: 图片路径
        :param thd: 相似度
        :return: 图片中心坐标
        '''
        img = self.GetCapture(x1, y1, x2, y2)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        template = cv2.imread(path, 0)
        th, tw = template.shape[::]
        rv = cv2.matchTemplate(img, template, 1)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(rv)
        if 1 - minVal >= thd:
            return minLoc[0] + tw / 2 + x1, minLoc[1] + th / 2 + y1
        else:
            return -1, -1

    def Hex_to_Rgb(self, hex):
        '''
        十六进制转RGB
        :param hex: 十六进制颜色值
        :return: RGB
        '''
        return np.array(tuple(int(hex[i:i + 2], 16) for i in (0, 2, 4)))

    def CmpColor(self, x, y, color, sim: float):
        '''
        比色
        :param x: X坐标
        :param y: Y坐标
        :param color: 十六进制颜色，可以从大漠直接获取
        :param sim: 相似偏移
        :return: 真或加
        '''
        img = self.GetCapture(x - 1, y - 1, x + 1, y + 1)
        img = np.array(img)
        img = img[1][1]
        color = self.Hex_to_Rgb(color)
        res = np.absolute(color - img)
        sim = int((1 - sim) * 255)
        return True if np.amax(res) <= sim else False

    def FindColor(self, x1, y1, x2, y2, des, sim: float):
        '''
        找色
        :param x1: 起点X
        :param y1: 起点Y
        :param x2: 终点X
        :param y2: 终点Y
        :param des: 十六进制颜色，可以从大漠直接获取
        :param sim: 相似偏移
        :return:
        '''
        img = self.GetCapture(x1, y1, x2, y2)
        img = np.array(img)
        res = np.absolute(img - self.Hex_to_Rgb(des))
        sim = int((1 - sim) * 255)
        res = np.argwhere(np.all(res <= sim, axis=2))
        res = res + (y1, x1)
        return res[:, [1, 0]]

    def GetColorNum(self, x1, y1, x2, y2, des, sim: float):
        '''
        获取颜色数量
        :param x1: 起点X
        :param y1: 起点Y
        :param x2: 终点X
        :param y2: 终点Y
        :param des: 十六进制颜色，可以从大漠直接获取
        :param sim: 相似偏移
        :return:
        '''
        return len(self.FindColor(x1, y1, x2, y2, des, sim))

    def FindMultColor(self, stax, stay, endx, endy, des):
        '''
        多点找色
        :param stax:
        :param stay:
        :param endx:
        :param endy:
        :param des: 大漠获取到的多点找色数据，偏色必须写上
        :return:
        '''
        w = endx - stax
        h = endy - stay
        img = pyautogui.screenshot(region=(stax, stay, w, h))
        img = np.array(img)
        rgby = []
        ps = []
        a = 0
        firstXY = []
        res = np.empty([0, 2])
        for i in des.split(','):
            rgb_y = i[-13:]
            r = int(rgb_y[0:2], 16)
            g = int(rgb_y[2:4], 16)
            b = int(rgb_y[4:6], 16)
            y = int(rgb_y[-2:])
            rgby.append([r, g, b, y])
        for i in range(1, len(des.split(','))):
            ps.append([int(des.split(',')[i].split('|')[0]), int(des.split(',')[i].split('|')[1])])
        for i in rgby:
            result = np.logical_and(abs(img[:, :, 0:1] - i[0]) < i[3], abs(img[:, :, 1:2] - i[1]) < i[3],
                                    abs(img[:, :, 2:3] - i[2]) < i[3])
            results = np.argwhere(np.all(result == True, axis=2)).tolist()
            if a == 0:
                firstXY = copy.deepcopy(results)
            else:
                nextnextXY = copy.deepcopy(results)
                for index in nextnextXY:
                    index[0] = int(index[0]) - ps[a - 1][1]
                    index[1] = int(index[1]) - ps[a - 1][0]
                q = set([tuple(t) for t in firstXY])
                w = set([tuple(t) for t in nextnextXY])
                matched = np.array(list(q.intersection(w)))
                res = np.append(res, matched, axis=0)
            a += 1
        unique, counts = np.unique(res, return_counts=True, axis=0)
        index = np.argmax(counts)
        re = unique[index] + (stay, stax)
        if np.max(counts) == len(des.split(',')) - 1:
            return np.flipud(re)
        return np.array([-1, -1])

    def FindPicEx(self, x1, y1, x2, y2, path, thd=0.9, MIN_MATCH_COUNT=8):
        '''
        全分辨率找图
        :param x1:
        :param y1:
        :param x2:
        :param y2:
        :param path:
        :param thd: 相似度
        :param MIN_MATCH_COUNT: 特征点数量
        :return:
        '''
        thd = thd - 0.2
        template = cv2.imread(path, 0)  # queryImage
        # target = cv2.imread('target.jpg', 0)  # trainImage
        target = self.GetCapture(x1, y1, x2, y2)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        # Initiate SIFT detector创建sift检测器
        sift = cv2.xfeatures2d.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(template, None)
        kp2, des2 = sift.detectAndCompute(target, None)
        # 创建设置FLANN匹配
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        # store all the good matches as per Lowe's ratio test.
        good = []
        # 舍弃大于0.7的匹配
        for m, n in matches:
            if m.distance < thd * n.distance:
                good.append(m)
        if len(good) > MIN_MATCH_COUNT:
            # 获取关键点的坐标
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            # 计算变换矩阵和MASK
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            h, w = template.shape
            # 使用得到的变换矩阵对原图像的四个角进行变换，获得在目标图像上对应的坐标
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            res = (dst[0] + dst[2]) / 2  # [[[ 39.11337  147.11575 ]] [[135.06624  255.12143 ]]
            return int(res[0][0]) + x1, int(res[0][1]) + y1
        else:
            return -1, -1

    def _FilterRec(self, res, loc):
        """ 对同一对象的多个框按位置聚类后，按置信度选最大的一个进行保留。
        :param res: 是 cv2.matchTemplate 返回值
        :param loc: 是 cv2.np.argwhere(res>threshold) 返回值
        :return: 返回保留的点的列表 pts
        """
        model = cluster.AffinityPropagation(damping=0.5, max_iter=100, convergence_iter=10, preference=-50).fit(loc)
        y_pred = model.labels_
        pts = []
        for i in set(y_pred):
            argj = loc[y_pred == i]
            argi = argj.T
            pt = argj[np.argmax(res[tuple(argi)])]
            pts.append(pt[::-1])
        return np.array(pts)

    def FindMultPic(self, x1, y1, x2, y2, path, thd):
        '''
        多目标找图
        :param x1:
        :param y1:
        :param x2:
        :param y2:
        :param path:
        :param thd: 相似度
        :return:
        '''
        target = self.GetCapture(x1, y1, x2, y2)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        template = cv2.imread(path, 0)
        w, h = template.shape[:2]
        res = cv2.matchTemplate(target, template, cv2.TM_CCOEFF_NORMED)
        loc = np.argwhere(res >= thd)
        if len(loc):
            resc = self._FilterRec(res, loc)
            return resc + (h / 2 + x1, w / 2 + y1)
        else:
            return [[-1, -1]]

    def FindPic_TM(self, x1, y1, x2, y2, path, thd):
        '''
        找透明图，透明色为黑色
        :param x1: 起点X
        :param y1: 起点Y
        :param x2: 终点X
        :param y2: 终点Y
        :param path: 图片路径
        :param thd: 相似度
        :return: 图片中心坐标
        '''
        img = self.GetCapture(x1, y1, x2, y2)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        template = cv2.imread(path)
        template2 = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(template2, 20, 255, cv2.THRESH_BINARY)
        th, tw = template.shape[:2]
        rv = cv2.matchTemplate(img, template, 1, mask=mask)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(rv)
        if 1 - minVal >= thd:
            return minLoc[0] + tw / 2 + x1, minLoc[1] + th / 2 + y1
        else:
            return -1, -1

    def StressShow(self, stax, stay, endx, endy, des, sim):
        '''
        保留选中颜色为白色，其他均为黑色
        :param stax:
        :param stay:
        :param endx:
        :param endy:
        :param des:
        :param sim:相似度，1为完全相同
        :return:
        '''
        sim = int((1 - sim) * 255)
        des = np.array(self.Hex_to_Rgb(des))
        img = self.GetCapture(stax, stay, endx, endy)
        mask = np.abs(img - des) <= sim
        new_mask = np.full(mask.shape, 0, dtype=np.uint8)
        new_mask[np.all(mask == True, axis=-1)] = [255, 255, 255]
        return new_mask


a = TuSe()
# b = a.StressShow(0, 0, 1820, 1080, 'ffffe1', 0.7)
b = a.FindPic_TM(0, 0, 1820, 1080,'234.bmp',0.9)
print(b)
pyautogui.moveTo(b[0],b[1])
# cv2.imshow('13', b)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
