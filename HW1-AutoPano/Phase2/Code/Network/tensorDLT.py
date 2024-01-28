import torch
import torch.nn as nn
import kornia

class TensorDLT(nn.Module):
    def __init__(self):
        super().__init__()
        # self.tensorDLT = kornia.geometry.transform.dlt()
        
    def tensorDLT(self, corners_a, preds):
        H = torch.tensor([], device=preds.device)
        for pred in preds:
            corners_b = corners_a + pred
            A, b = [], []
            for i in range(0, 8, 2):
                Ai = [
                    [0, 0, 0, -corners_a[i], -corners_a[i + 1], -1, corners_b[i + 1] * corners_a[i], corners_b[i + 1] * corners_a[i + 1]],
                    [corners_a[i], corners_a[i + 1], 1, 0, 0, 0, -corners_b[i] * corners_a[i], -corners_b[i] * corners_a[i + 1]]
                ]
                A.extend(Ai)

                bi = [-corners_b[i + 1], -corners_b[i]]
                b.extend(bi)
            A = torch.tensor(A, device=pred.device)
            b = torch.tensor(b, device=pred.device)
            
            h = torch.linalg.pinv(A)@ b
            
            print(h)
            H = torch.cat(H,h.reshape(1,-1), axis=0)
        
    def forward(self, corners_a, preds):
        return self.tensorDLT(corners_a, preds)



def test_tensorDLT():
    test_L = torch.tensor([
                            [
                                [-23,   -18, -29 , 29 , -5 , -22 , 5 ,  32],
                                [-213, -181, -219, 219, -15, -212, 15, 312],
                                [-223, -281, -229, 229, -25, -222, 25, 322],
                                [-233, -381, -239, 239, -35, -232, 35, 322],
                            ],
                            [
                                [-1  , -8  , -91 , 9 , -65  , -4 , -5,  37],
                                [-233, -118, -229, 291, -513, -21, 52, -20],
                                [-243, -128, -239, 292, -523, -22, 62, -30],
                                [-253, -138, -249, 293, -533, -23, 72, -40]
                            ]
                        ],dtype=torch.float64)
    actual_H4pt_list = [
                            [
                                [0,   0, 128 , 0 , 0 , 128 , 128 ,  128],
                                [-213, -181, -219, 219, -15, -212, 15, 312],
                            ],
                            [
                                [-23, -18, 99, 29, -5, 106, 133, 160],
                                [-233, -118, -229, 291, -513, -21, 52, -20],
                            ]

                    ]

    import numpy as np
    H4pt_src = np.array(actual_H4pt_list[0][0]).reshape(4,2)
    print(f'src:{H4pt_src}')
    H4pt_dst = np.array(actual_H4pt_list[1][0]).reshape(4,2)
    print(f'dst:{H4pt_dst}')

    import cv2
    H_cv2,_ = cv2.findHomography(H4pt_src,H4pt_dst)

    H4pt_torch = torch.tensor(actual_H4pt_list,dtype=torch.float64)

    # test_img = torch.zeros(size=(2,128,128))
    H_torch = tensorDLT(H4pt_torch[0],H4pt_torch[1])
    print(H_cv2)
    print(H_torch[0])