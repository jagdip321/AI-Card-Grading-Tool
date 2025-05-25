import cv2
import numpy as np

def detect_card_color_based(image_path, show_steps=False):
    image = cv2.imread(image_path)
    orig = image.copy()
    height, width = image.shape[:2]

    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for purple background (tuned for your image)
    lower_purple = np.array([120, 40, 40])
    upper_purple = np.array([160, 255, 255])
    mask = cv2.inRange(hsv, lower_purple, upper_purple)

    # Invert the mask to isolate non-purple (i.e., the card)
    card_mask = cv2.bitwise_not(mask)

    # Morphological cleaning
    kernel = np.ones((5,5), np.uint8)
    card_mask = cv2.morphologyEx(card_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(card_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"error": "Card not detected"}

    # Assume largest non-purple area is the card
    card_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(card_contour)

    # Margins
    left = x
    top = y
    right = width - (x + w)
    bottom = height - (y + h)

    # Centering score
    def to_ratio(a, b): return min(a, b) / (a + b + 1e-5)

    h_ratio = to_ratio(left, right)
    v_ratio = to_ratio(top, bottom)

    def grade(r):
        if r >= 0.45: return 10
        elif r >= 0.40: return 9
        elif r >= 0.35: return 8
        elif r >= 0.30: return 7
        else: return 6

    h_grade = grade(h_ratio)
    v_grade = grade(v_ratio)

    if show_steps:
        annotated = orig.copy()

        # Detected card border (green)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 4)

        # Determine average margin (to simulate ideal centering)
        avg_h_margin = int((left + right) / 2)
        avg_v_margin = int((top + bottom) / 2)

        # Simulated perfectly centered box (same size as detected card, but centered in image)
        center_x = (width - w) // 2
        center_y = (height - h) // 2
        cv2.rectangle(
            annotated,
            (center_x, center_y),
            (center_x + w, center_y + h),
            (0, 0, 255),
            2
        )

        # Grade label
        cv2.putText(
            annotated,
            f"PSA Centering Grade: {min(h_grade, v_grade)}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 255),
            3
        )

        # Resize for display
        display = cv2.resize(annotated, (800, int(800 * height / width)))
        cv2.namedWindow("Detected Card", cv2.WINDOW_NORMAL)
        cv2.imshow("Detected Card", display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



    return {
        "margins": {"left": left, "right": right, "top": top, "bottom": bottom},
        "center_ratios": {"horizontal": h_ratio, "vertical": v_ratio},
        "grades": {
            "horizontal": h_grade,
            "vertical": v_grade,
            "final_centering_grade": min(h_grade, v_grade)
        }
    }

# Run the analysis
if __name__ == "__main__":
    path = "BackCard1.jpeg"
    result = detect_card_color_based(path, show_steps=True)
    print(result)
    print(f"\nFinal Centering Grade: PSA {result['grades']['final_centering_grade']}")
    print(f"Horizontal Centering: {result['center_ratios']['horizontal']*100:.1f}%")
    print(f"Vertical Centering: {result['center_ratios']['vertical']*100:.1f}%")
    print("Margins (px):", result['margins'])

