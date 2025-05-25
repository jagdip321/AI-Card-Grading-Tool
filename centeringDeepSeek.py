import cv2
import numpy as np
import os
import sys

def enhance_contrast(image):
    """Improve contrast for better edge detection"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced

def sharpen_image(image):
    """Enhance edges in the image"""
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def preprocess_image(image):
    """Prepare image for contour detection"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(blurred)
    sharpened = sharpen_image(enhanced)
    edged = cv2.Canny(sharpened, 30, 150)
    return edged

def order_points(pts):
    """Arrange coordinates in consistent order (top-left, top-right, bottom-right, bottom-left)"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left has smallest sum
    rect[2] = pts[np.argmax(s)]  # bottom-right has largest sum
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right has smallest difference
    rect[3] = pts[np.argmax(diff)]  # bottom-left has largest difference
    return rect

def detect_card(image):
    """Improved card detection that works for both front and back faces"""
    # Enhanced preprocessing
    enhanced = enhance_contrast(image)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Combined edge detection
    edged1 = cv2.Canny(blurred, 30, 150)
    edged2 = cv2.Canny(gray, 50, 150)
    edged = cv2.bitwise_or(edged1, edged2)
    
    # Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        if len(approx) == 4 and cv2.contourArea(contour) > 50000:
            return order_points(approx.reshape(4, 2))
    
    # Fallback method for difficult cards
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
        if len(approx) == 4 and cv2.contourArea(contour) > 30000:
            return order_points(approx.reshape(4, 2))
    
    return None


def extract_card(image, card_contour):
    """Extract and straighten the card using perspective transform"""
    width = max(np.linalg.norm(card_contour[0] - card_contour[1]),
                np.linalg.norm(card_contour[2] - card_contour[3]))
    height = max(np.linalg.norm(card_contour[0] - card_contour[3]),
                 np.linalg.norm(card_contour[1] - card_contour[2]))
    
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(card_contour, dst)
    warped = cv2.warpPerspective(image, M, (int(width), int(height)))
    return warped

def detect_face_borders(card_image):
    """Improved face border detection that works for both front and back"""
    # Convert to grayscale and enhance
    gray = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    # Filter by area and rectangular shape
    min_area = card_image.shape[0] * card_image.shape[1] * 0.4
    valid_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
            
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        if len(approx) == 4:
            valid_contours.append(approx)
    
    if not valid_contours:
        return None
    
    # Select the most central contour
    card_center = np.array([card_image.shape[1]/2, card_image.shape[0]/2])
    best_contour = min(valid_contours, 
                      key=lambda c: np.linalg.norm(card_center - cv2.minEnclosingCircle(c)[0]))
    
    return order_points(best_contour.reshape(4, 2))


def calculate_centering(card_shape, inner_rect):
    """Calculate border widths and centering ratios"""
    card_height, card_width = card_shape[:2]
    
    # Get bounding rect of inner rectangle
    x, y, w, h = cv2.boundingRect(inner_rect.astype(int))
    
    # Calculate borders
    left = x
    right = card_width - (x + w)
    top = y
    bottom = card_height - (y + h)
    
    # Calculate PSA standard ratios (min/max)
    horizontal_ratio = min(left, right) / max(left, right)
    vertical_ratio = min(top, bottom) / max(top, bottom)
    
    return {
        'left': left,
        'right': right,
        'top': top,
        'bottom': bottom,
        'horizontal_ratio': horizontal_ratio,
        'vertical_ratio': vertical_ratio,
        'card_width': card_width,
        'card_height': card_height
    }

def estimate_psa_grade(metrics):
    """Estimate PSA grade based on centering ratios"""
    min_ratio = min(metrics['horizontal_ratio'], metrics['vertical_ratio'])
    
    if min_ratio >= 0.75: return "PSA 10 - Gem Mint"
    elif min_ratio >= 0.70: return "PSA 9 - Mint"
    elif min_ratio >= 0.65: return "PSA 8 - Near Mint/Mint"
    elif min_ratio >= 0.60: return "PSA 7 - Near Mint"
    elif min_ratio >= 0.55: return "PSA 6 - Excellent"
    elif min_ratio >= 0.50: return "PSA 5 - Very Good/Excellent"
    else: return "PSA 4 or below - Poor Centering"

def display_results(original_image, card_contour, card_image, inner_rect, metrics, grade):
    """Display original image with card detection and analysis results"""
    # Draw on original image
    orig_display = original_image.copy()
    cv2.drawContours(orig_display, [card_contour.astype(int)], -1, (0, 255, 0), 3)
    cv2.putText(orig_display, "Detected Card", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw on card image
    card_display = card_image.copy()
    if inner_rect is not None:
        cv2.polylines(card_display, [inner_rect.astype(int)], True, (0, 0, 255), 2)
    
    # Add text with measurements
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_offset = 30
    texts = [
        f"Left/Right: {metrics['left']:.1f}/{metrics['right']:.1f} px",
        f"Top/Bottom: {metrics['top']:.1f}/{metrics['bottom']:.1f} px",
        f"Horizontal: {metrics['horizontal_ratio']:.3f}",
        f"Vertical: {metrics['vertical_ratio']:.3f}",
        f"Grade: {grade}"
    ]
    
    for i, text in enumerate(texts):
        cv2.putText(card_display, text, (10, y_offset + i*30), font, 0.7, (0,0,0), 4)
        cv2.putText(card_display, text, (10, y_offset + i*30), font, 0.7, (255,255,255), 2)
    
    # Show both images
    cv2.namedWindow("1. Original Image with Card Detected", cv2.WINDOW_NORMAL)
    cv2.imshow("1. Original Image with Card Detected", orig_display)
    
    cv2.namedWindow("2. Card Analysis", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("2. Card Analysis", 600, 800)
    cv2.imshow("2. Card Analysis", card_display)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        possible_images = [f for f in os.listdir() if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if possible_images:
            image_path = possible_images[0]
            print(f"Using image: {image_path}")
        else:
            print("Usage: python psa_grading.py <image_path>")
            return
    
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    # 1. Detect card
    card_contour = detect_card(original_image)
    if card_contour is None:
        print("Card detection failed. Try:")
        print("- Better lighting and plain background")
        print("- Ensure card fills most of the image")
        print("- Hold card straight (not at angle)")
        return
    
    # 2. Extract card
    card_image = extract_card(original_image, card_contour)
    
    # 3. Detect face borders
    inner_rect = detect_face_borders(card_image)
    if inner_rect is None:
        print("Face border detection failed. Using fallback method.")
        h, w = card_image.shape[:2]
        margin = 0.07  # Slightly larger margin for front faces
        inner_rect = np.array([
            [w*margin, h*margin],
            [w*(1-margin), h*margin],
            [w*(1-margin), h*(1-margin)],
            [w*margin, h*(1-margin)]
        ], dtype="float32")
    
    # 4. Calculate centering
    metrics = calculate_centering(card_image.shape, inner_rect)
    grade = estimate_psa_grade(metrics)
    
    # 5. Display results
    display_results(original_image, card_contour, card_image, inner_rect, metrics, grade)
    
    print("\n=== Grading Results ===")
    print(f"Dimensions: {metrics['card_width']}x{metrics['card_height']}px")
    print(f"Borders - Left: {metrics['left']:.1f}px, Right: {metrics['right']:.1f}px")
    print(f"          Top: {metrics['top']:.1f}px, Bottom: {metrics['bottom']:.1f}px")
    print(f"Ratios - Horizontal: {metrics['horizontal_ratio']:.3f}, Vertical: {metrics['vertical_ratio']:.3f}")
    print(f"Grade: {grade}")


if __name__ == "__main__":
    main()