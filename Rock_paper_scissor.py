import random

import cv2
import cvzone
import mediapipe as mp
import time

class HandDetector:
    """
    Finds Hands using the mediapipe library. Exports the landmarks
    in pixel format. Adds extra functionalities like finding how
    many fingers are up. Also
    provides bounding box info of the hand found.
    """

    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, minTrackCon=0.5):
        """
        :param mode: In static mode, detection is done on each image: slower
        :param maxHands: Maximum number of hands to detect
        :param detectionCon: Minimum Detection Confidence Threshold
        :param minTrackCon: Minimum Tracking Confidence Threshold
        """
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.minTrackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.fingers = []
        self.lmList = []

    def findHands(self, img, draw=True, flipType=True):
        """
        Finds hands in a BGR image.
        :param img: Image to find the hands in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        allHands = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {}
                ## lmList
                mylmList = []
                xList = []
                yList = []
                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    mylmList.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)

                ## bbox
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0] + (bbox[2] // 2), \
                         bbox[1] + (bbox[3] // 2)

                myHand["lmList"] = mylmList
                myHand["bbox"] = bbox
                myHand["center"] = (cx, cy)

                if flipType:
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Left"
                    else:
                        myHand["type"] = "Right"
                else:
                    myHand["type"] = handType.classification[0].label
                allHands.append(myHand)

                ## draw
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
                    cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                                  (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                                  (255, 0, 255), 2)
                    cv2.putText(img, myHand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                                2, (255, 0, 255), 2)
        if draw:
            return allHands, img
        else:
            return allHands

    def fingersUp(self, myHand):
        """
        Finds how many fingers are open and returns in a list.
        Considers left and right hands separately
        :return: List of which fingers are up
        """
        myHandType = myHand["type"]
        myLmList = myHand["lmList"]       
        if self.results.multi_hand_landmarks:
            fingers = []
            # Thumb
            if myHandType == "Right":
                if myLmList[self.tipIds[0]][0] > myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if myLmList[self.tipIds[0]][0] < myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # 4 Fingers
            for id in range(1, 5):
                print(myLmList[self.tipIds[id]][1])
                print(myLmList[self.tipIds[id] - 2][1])
                if myLmList[self.tipIds[id]][1] < myLmList[self.tipIds[id] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
                    
            print(fingers)
        return fingers

quay_Video = cv2.VideoCapture(0)  # khai báo camera với camera id là 0 vì có thể có nhiều camera
anh_nen = cv2.imread("D:/AIP391/Rock---Paper---Scissor/Du_Lieu/BG.png")
quay_Video.set(3, 640)  # thiết lập chiều rộng của cửa sổ camera
quay_Video.set(4, 480)  # thiết lập chiều cao của cửa sổ camera

nhan_dien_ban_tay = HandDetector(maxHands=1)  # khởi tạo object nhận diện bàn tay
# và số bàn tay tối đa nhận diện là 1 và để giá trị
# confidence value là default = 0.5

dem_thoi_gian = 0  # khỏi tạo biến đếm thời gian để bắt đầu game sau khi đến thời gian quy định
trang_thai_ket_qua = False
Bat_dau_game = False
Ti_So = [0, 0] # [AI, Nguoi_choi]
while True:
    thanh_cong, anh = quay_Video.read()  # doc anh tu camera
    # biến thanh_cong kieu boolean de xac nhan da nhan dien chưa
    anh_nen = cv2.imread("D:/AIP391/Rock---Paper---Scissor/Du_Lieu/BG.png")  # doc anh nen de lam giao dien tro choi
    anh_Scaled = cv2.resize(anh, (0, 0), None, 0.875, 0.875)  # ti le lại ảnh trong video sao cho chiều cao
    # bằng chiều cao của ô trò chơi sẽ để ảnh video vào(cắt chiều cao)
    anh_Scaled = anh_Scaled[:, 80:480]  # ảnh là ma trận,
    # giờ ta sẽ cắt từ 2 bên cạnh của ảnh(chỉ cắt chiều ngang nên bỏ trống tham số chiều cao)
    # Tìm tay trong khung hình video
    tay, nguon_de_nhan_dien_tay = nhan_dien_ban_tay.findHands(
        anh_Scaled)  # có vẽ hình nhận diện bàn tay trong khung hình video
    # lưu ý phải nhận diện xong bàn tay mới cho vào phần khung ảnh trò chơi
    # nếu cho trước nó sẽ nhận diện sau và không cho vào phần đồ họa cần hiển thị lên để chơi
    if Bat_dau_game:

        if trang_thai_ket_qua is False:  # chưa đạt đến thời gian bắt đầu trò chơi quy định
            dem_thoi_gian = time.time() - thoi_gian_ban_dau  # thoi gian dem bang hien tai sau an 's' - ban dau
            cv2.putText(anh_nen, str(int(dem_thoi_gian)), (605, 435), cv2.FONT_HERSHEY_PLAIN, 6, (255, 0, 255), 4)
            # đưa thời gian dem len man hinh choi bien thoi gian ep kiểu về int rồi về str, các tham số
            # lần lượt là vị trí đặt vào trên man hinh(anh_nen), kiểu font, tỉ lệ độ lớn, màu sắc(tím)
            # và độ dày(thickness = 4)
            if dem_thoi_gian > 3:  # không nên để >=3 vì đang làm việc với thời gian là số thực
                # nên nhiều khi chạy vòng lặp nó sẽ không vào 3 làm ta thêm một bước
                # so sánh = không cần thiết hao tốn tài nguyên và thời gian
                trang_thai_ket_qua = True  # dừng việc đếm thời gian do biến kết quả đã về True
                dem_thoi_gian = 0  # cho biến đêm thời gian về 0 cho lần chơi tiếp theo
                if tay:  # nếu nhận diện được tay thì thực hiện tiếp
                    tay_nguoi_choi_ra = None  # khởi tạo biến tay người chơi bằng none
                    ban_tay = tay[0]  # khởi tạo đối tượng bàn tay chính
                    dem_ngon_tay = nhan_dien_ban_tay.fingersUp(ban_tay)  # tạo ra một mảng 5 phần tử đại diện cho 5 ngón tay,
                    # mỗi phần tử có giá trị là 0 hoặc 1 để phân biệt ngón tay có giơ lên hay cụp xuống
                    if dem_ngon_tay == [0, 0, 0, 0, 0]:
                        tay_nguoi_choi_ra = 1
                    if dem_ngon_tay == [1, 1, 1, 1, 1]:
                        tay_nguoi_choi_ra = 2
                    if dem_ngon_tay == [0, 1, 1, 0, 0]:
                        tay_nguoi_choi_ra = 3

                    random_anh_cho_AI = random.randint(1, 3)  # random 1 -> 3 tương đương cho 3 ảnh và 3 khả năng ra
                    anh_cho_AI = cv2.imread(f"Du_Lieu/{random_anh_cho_AI}.png", cv2.IMREAD_UNCHANGED)   # đọc ảnh thành giá trị(thêm f ở đầu)
                    # sau đó gán thành biến và nhớ thêm cv2.IMREAD_UNCHANGED viết hoa
                    # để sử dụng overlay(lấp, đè lên ảnh nền) nó sẽ mất giá trị hình ảnh
                    anh_nen = cvzone.overlayPNG(anh_nen, anh_cho_AI, (149, 310))
                    print(tay_nguoi_choi_ra)

                    #Người Chơi Thắng
                    if(tay_nguoi_choi_ra == 1 and random_anh_cho_AI == 3) or \
                            (tay_nguoi_choi_ra == 2 and random_anh_cho_AI == 1) or \
                            (tay_nguoi_choi_ra == 3 and random_anh_cho_AI == 2):
                        Ti_So[1] += 1
                    #AI chiến thắng
                    if (tay_nguoi_choi_ra == 3 and random_anh_cho_AI == 1) or \
                            (tay_nguoi_choi_ra == 1 and random_anh_cho_AI == 2) or \
                            (tay_nguoi_choi_ra == 2 and random_anh_cho_AI == 3):
                        Ti_So[0] += 1

    anh_nen[234:654, 795:1195] = anh_Scaled  # đặt ảnh video vào ô người chơi và kích thước đặt vào vào khớp với
    # kích thước đã định trước, 233:653 là vị trí của 2 hàng trong ma trận ảnh,
    # 795:1195 là vị trí 2 cột trong ma trận ảnh --> hàng trước cột sau
    # cv2.imshow("Day La Video", anh)  # tao cua so video
    if trang_thai_ket_qua:
        anh_nen = cvzone.overlayPNG(anh_nen, anh_cho_AI, (149, 310))    # đặt lại dòng lưu kết quả trả ra của AI
        # ngoài các lệnh điều kiện để nó có thể giữ trạng thái hiển thị
        # vì đặt trong điều kiện chỉ thực hiện 1 lần(hiện lên rồi biến mất ngay) trong 1 lần chơi
    cv2.putText(anh_nen, str(Ti_So[0]), (410, 215), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 6)
    cv2.putText(anh_nen, str(Ti_So[1]), (1112, 215), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 6)
    cv2.imshow("Rock-Paper-Scissor Game", anh_nen)  # tao cua so tro choi voi khung nen duoc nap vao
    phim_bat_dau = cv2.waitKey(1)  # độ trễ sau khi nhấn là 1 millis
    if phim_bat_dau == ord('s'):  # nút bắt đầu trò chơi là s (khởi tạo 1 flag)
        Bat_dau_game = True
        thoi_gian_ban_dau = time.time()  # khởi tạo thời gian ban đầu
        trang_thai_ket_qua = False  # quan trọng reset lại trạng thái ket qua để chuẩn bị cho những lần chơi tiếp theo khi ấn s
