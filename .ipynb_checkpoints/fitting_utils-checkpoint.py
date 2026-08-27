import numpy as np
import cv2

def shift_rotate(model, dx, dy, theta_deg, order=1):
    h, w = model.shape
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), theta_deg, 1.0)
    M[0, 2] += dx
    M[1, 2] += dy
    interp = {0: cv2.INTER_NEAREST, 1: cv2.INTER_LINEAR, 3: cv2.INTER_CUBIC}[order]
    return cv2.warpAffine(
        model.astype(np.float32), M, (w, h),
        flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,)

def asymmetric_huber(r, delta=1.0, lam=3.0):
    abs_r = np.abs(r)

    huber = np.where(
        abs_r <= delta,
        0.5 * r**2,
        delta * (abs_r - 0.5 * delta))

    return np.where(r < 0, lam * huber, huber)

def asymmetric_loss(params, data, model, error,mask, delta = .10, lam=2.5):

    dx, dy, theta, amp = params

    shifted_model = shift_rotate( np.asarray(model), dx=dx, dy=dy, theta_deg=theta)

    residual = data - shifted_model*amp

    # Weight by uncertainty
    r = residual[mask] / error[mask]

    # Penalize oversubtraction more strongly
    loss = asymmetric_huber(r, delta = delta, lam = lam)
    
    return np.sum(loss)
