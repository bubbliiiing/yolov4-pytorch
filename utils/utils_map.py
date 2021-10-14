import glob
import json
import math
import operator
import os
import shutil
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

'''
    0,0 ------> x (width)
     |
     |  (Left,Top)
     |      *_________
     |      |         |
            |         |
     y      |_________|
  (height)            *
                (Right,Bottom)
'''

def log_average_miss_rate(precision, fp_cumsum, num_images):
    """
        log-average miss rate:
            Calculated by averaging miss rates at 9 evenly spaced FPPI points
            between 10e-2 and 10e0, in log-space.

        output:
                lamr | log-average miss rate
                mr | miss rate
                fppi | false positives per image

        references:
            [1] Dollar, Piotr, et al. "Pedestrian Detection: An Evaluation of the
               State of the Art." Pattern Analysis and Machine Intelligence, IEEE
               Transactions on 34.4 (2012): 743 - 761.
    """

    if precision.size == 0:
        lamr = 0
        mr = 1
        fppi = 0
        return lamr, mr, fppi

    fppi = fp_cumsum / float(num_images)
    mr = (1 - precision)

    fppi_tmp = np.insert(fppi, 0, -1.0)
    mr_tmp = np.insert(mr, 0, 1.0)

    ref = np.logspace(-2.0, 0.0, num = 9)
    for i, ref_i in enumerate(ref):
        j = np.where(fppi_tmp <= ref_i)[-1][-1]
        ref[i] = mr_tmp[j]

    lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))

    return lamr, mr, fppi

"""
 throw error and exit
"""
def error(msg):
    print(msg)
    sys.exit(0)

"""
 check if the number is a float between 0.0 and 1.0
"""
def is_float_between_0_and_1(value):
    try:
        val = float(value)
        if val > 0.0 and val < 1.0:
            return True
        else:
            return False
    except ValueError:
        return False

"""
 Calculate the AP given the recall and precision array
    1st) We compute a version of the measured precision/recall curve with
         precision monotonically decreasing
    2nd) We compute the AP as the area under this curve by numerical integration.
"""
def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre


"""
 Convert the lines of a file to a list
"""
def file_lines_to_list(path):
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content

"""
 Draws text in image
"""
def draw_text_in_image(img, text, pos, color, line_width):
    font = cv2.FONT_HERSHEY_PLAIN
    fontScale = 1
    lineType = 1
    bottomLeftCornerOfText = pos
    cv2.putText(img, text,
            bottomLeftCornerOfText,
            font,
            fontScale,
            color,
            lineType)
    text_width, _ = cv2.getTextSize(text, font, fontScale, lineType)[0]
    return img, (line_width + text_width)

"""
 Plot - adjust axes
"""
def adjust_axes(r, t, fig, axes):
    # get text width for re-scaling
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    # get axis width in inches
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    # get axis limit
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1]*propotion])

"""
 Draw plot using Matplotlib
"""
def draw_plot_func(dictionary, n_classes, window_title, plot_title, x_label, output_path, to_show, plot_color, true_p_bar):
    # sort the dictionary by decreasing value, into a list of tuples
    sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1))
    # unpacking the list of tuples into two lists
    sorted_keys, sorted_values = zip(*sorted_dic_by_value)
    # 
    if true_p_bar != "":
        """
         Special case to draw in:
            - green -> TP: True Positives (object detected and matches ground-truth)
            - red -> FP: False Positives (object detected but does not match ground-truth)
            - orange -> FN: False Negatives (object not detected but present in the ground-truth)
        """
        fp_sorted = []
        tp_sorted = []
        for key in sorted_keys:
            fp_sorted.append(dictionary[key] - true_p_bar[key])
            tp_sorted.append(true_p_bar[key])
        plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='False Positive')
        plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen', label='True Positive', left=fp_sorted)
        # add legend
        plt.legend(loc='lower right')
        """
         Write number on side of bar
        """
        fig = plt.gcf() # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            fp_val = fp_sorted[i]
            tp_val = tp_sorted[i]
            fp_str_val = " " + str(fp_val)
            tp_str_val = fp_str_val + " " + str(tp_val)
            # trick to paint multicolor with offset:
            # first paint everything and then repaint the first number
            t = plt.text(val, i, tp_str_val, color='forestgreen', va='center', fontweight='bold')
            plt.text(val, i, fp_str_val, color='crimson', va='center', fontweight='bold')
            if i == (len(sorted_values)-1): # largest bar
                adjust_axes(r, t, fig, axes)
    else:
        plt.barh(range(n_classes), sorted_values, color=plot_color)
        """
         Write number on side of bar
        """
        fig = plt.gcf() # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            str_val = " " + str(val) # add a space before
            if val < 1.0:
                str_val = " {0:.2f}".format(val)
            t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
            # re-set axes to show number inside the figure
            if i == (len(sorted_values)-1): # largest bar
                adjust_axes(r, t, fig, axes)
    # set window title
    fig.canvas.set_window_title(window_title)
    # write classes in y axis
    tick_font_size = 12
    plt.yticks(range(n_classes), sorted_keys, fontsize=tick_font_size)
    """
     Re-scale height accordingly
    """
    init_height = fig.get_figheight()
    # comput the matrix height in points and inches
    dpi = fig.dpi
    height_pt = n_classes * (tick_font_size * 1.4) # 1.4 (some spacing)
    height_in = height_pt / dpi
    # compute the required figure height 
    top_margin = 0.15 # in percentage of the figure height
    bottom_margin = 0.05 # in percentage of the figure height
    figure_height = height_in / (1 - top_margin - bottom_margin)
    # set new height
    if figure_height > init_height:
        fig.set_figheight(figure_height)

    # set plot title
    plt.title(plot_title, fontsize=14)
    # set axis titles
    # plt.xlabel('classes')
    plt.xlabel(x_label, fontsize='large')
    # adjust size of window
    fig.tight_layout()
    # save the plot
    fig.savefig(output_path)
    # show image
    if to_show:
        plt.show()
    # close the plot
    plt.close()

def get_map(MINOVERLAP, draw_plot, path = './map_out'):
    GT_PATH             = os.path.join(path, 'ground-truth')
    DR_PATH             = os.path.join(path, 'detection-results')
    IMG_PATH            = os.path.join(path, 'images-optional')
    TEMP_FILES_PATH     = os.path.join(path, '.temp_files')
    RESULTS_FILES_PATH  = os.path.join(path, 'results')

    show_animation = True
    if os.path.exists(IMG_PATH): 
        for dirpath, dirnames, files in os.walk(IMG_PATH):
            if not files:
                show_animation = False
    else:
        show_animation = False

    if not os.path.exists(TEMP_FILES_PATH):
        os.makedirs(TEMP_FILES_PATH)
        
    if os.path.exists(RESULTS_FILES_PATH):
        shutil.rmtree(RESULTS_FILES_PATH)
    if draw_plot:
        os.makedirs(os.path.join(RESULTS_FILES_PATH, "AP"))
        os.makedirs(os.path.join(RESULTS_FILES_PATH, "F1"))
        os.makedirs(os.path.join(RESULTS_FILES_PATH, "Recall"))
        os.makedirs(os.path.join(RESULTS_FILES_PATH, "Precision"))
    if show_animation:
        os.makedirs(os.path.join(RESULTS_FILES_PATH, "images", "detections_one_by_one"))

    ground_truth_files_list = glob.glob(GT_PATH + '/*.txt')
    if len(ground_truth_files_list) == 0:
        error("Error: No ground-truth files found!")
    ground_truth_files_list.sort()
    gt_counter_per_class     = {}
    counter_images_per_class = {}

    for txt_file in ground_truth_files_list:
        file_id     = txt_file.split(".txt", 1)[0]
        file_id     = os.path.basename(os.path.normpath(file_id))
        temp_path   = os.path.join(DR_PATH, (file_id + ".txt"))
        if not os.path.exists(temp_path):
            error_msg = "Error. File not found: {}\n".format(temp_path)
            error(error_msg)
        lines_list      = file_lines_to_list(txt_file)
        bounding_boxes  = []
        is_difficult    = False
        already_seen_classes = []
        for line in lines_list:
            try:
                if "difficult" in line:
                    class_name, left, top, right, bottom, _difficult = line.split()
                    is_difficult = True
                else:
                    class_name, left, top, right, bottom = line.split()
            except:
                if "difficult" in line:
                    line_split  = line.split()
                    _difficult  = line_split[-1]
                    bottom      = line_split[-2]
                    right       = line_split[-3]
                    top         = line_split[-4]
                    left        = line_split[-5]
                    class_name  = ""
                    for name in line_split[:-5]:
                        class_name += name + " "
                    class_name  = class_name[:-1]
                    is_difficult = True
                else:
                    line_split  = line.split()
                    bottom      = line_split[-1]
                    right       = line_split[-2]
                    top         = line_split[-3]
                    left        = line_split[-4]
                    class_name  = ""
                    for name in line_split[:-4]:
                        class_name += name + " "
                    class_name = class_name[:-1]

            bbox = left + " " + top + " " + right + " " + bottom
            if is_difficult:
                bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False, "difficult":True})
                is_difficult = False
            else:
                bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False})
                if class_name in gt_counter_per_class:
                    gt_counter_per_class[class_name] += 1
                else:
                    gt_counter_per_class[class_name] = 1

                if class_name not in already_seen_classes:
                    if class_name in counter_images_per_class:
                        counter_images_per_class[class_name] += 1
                    else:
                        counter_images_per_class[class_name] = 1
                    already_seen_classes.append(class_name)

        with open(TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    gt_classes  = list(gt_counter_per_class.keys())
    gt_classes  = sorted(gt_classes)
    n_classes   = len(gt_classes)

    dr_files_list = glob.glob(DR_PATH + '/*.txt')
    dr_files_list.sort()
    for class_index, class_name in enumerate(gt_classes):
        bounding_boxes = []
        for txt_file in dr_files_list:
            file_id = txt_file.split(".txt",1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))
            temp_path = os.path.join(GT_PATH, (file_id + ".txt"))
            if class_index == 0:
                if not os.path.exists(temp_path):
                    error_msg = "Error. File not found: {}\n".format(temp_path)
                    error(error_msg)
            lines = file_lines_to_list(txt_file)
            for line in lines:
                try:
                    tmp_class_name, confidence, left, top, right, bottom = line.split()
                except:
                    line_split      = line.split()
                    bottom          = line_split[-1]
                    right           = line_split[-2]
                    top             = line_split[-3]
                    left            = line_split[-4]
                    confidence      = line_split[-5]
                    tmp_class_name  = ""
                    for name in line_split[:-5]:
                        tmp_class_name += name + " "
                    tmp_class_name  = tmp_class_name[:-1]

                if tmp_class_name == class_name:
                    bbox = left + " " + top + " " + right + " " +bottom
                    bounding_boxes.append({"confidence":confidence, "file_id":file_id, "bbox":bbox})

        bounding_boxes.sort(key=lambda x:float(x['confidence']), reverse=True)
        with open(TEMP_FILES_PATH + "/" + class_name + "_dr.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    sum_AP = 0.0
    ap_dictionary = {}
    lamr_dictionary = {}
    with open(RESULTS_FILES_PATH + "/results.txt", 'w') as results_file:
        results_file.write("# AP and precision/recall per class\n")
        count_true_positives = {}

        for class_index, class_name in enumerate(gt_classes):
            count_true_positives[class_name] = 0
            dr_file = TEMP_FILES_PATH + "/" + class_name + "_dr.json"
            dr_data = json.load(open(dr_file))

            nd          = len(dr_data)
            tp          = [0] * nd
            fp          = [0] * nd
            score       = [0] * nd
            score05_idx = 0
            for idx, detection in enumerate(dr_data):
                file_id     = detection["file_id"]
                score[idx]  = float(detection["confidence"])
                if score[idx] > 0.5:
                    score05_idx = idx

                if show_animation:
                    ground_truth_img = glob.glob1(IMG_PATH, file_id + ".*")
                    if len(ground_truth_img) == 0:
                        error("Error. Image not found with id: " + file_id)
                    elif len(ground_truth_img) > 1:
                        error("Error. Multiple image with id: " + file_id)
                    else:
                        img = cv2.imread(IMG_PATH + "/" + ground_truth_img[0])
                        img_cumulative_path = RESULTS_FILES_PATH + "/images/" + ground_truth_img[0]
                        if os.path.isfile(img_cumulative_path):
                            img_cumulative = cv2.imread(img_cumulative_path)
                        else:
                            img_cumulative = img.copy()
                        bottom_border = 60
                        BLACK = [0, 0, 0]
                        img = cv2.copyMakeBorder(img, 0, bottom_border, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)

                gt_file             = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
                ground_truth_data   = json.load(open(gt_file))
                ovmax       = -1
                gt_match    = -1
                bb          = [float(x) for x in detection["bbox"].split()]
                for obj in ground_truth_data:
                    if obj["class_name"] == class_name:
                        bbgt    = [ float(x) for x in obj["bbox"].split() ]
                        bi      = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
                        iw      = bi[2] - bi[0] + 1
                        ih      = bi[3] - bi[1] + 1
                        if iw > 0 and ih > 0:
                            ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                            + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                            ov = iw * ih / ua
                            if ov > ovmax:
                                ovmax = ov
                                gt_match = obj

                if show_animation:
                    status = "NO MATCH FOUND!" 
                    
                min_overlap = MINOVERLAP
                if ovmax >= min_overlap:
                    if "difficult" not in gt_match:
                        if not bool(gt_match["used"]):
                            tp[idx] = 1
                            gt_match["used"] = True
                            count_true_positives[class_name] += 1
                            with open(gt_file, 'w') as f:
                                    f.write(json.dumps(ground_truth_data))
                            if show_animation:
                                status = "MATCH!"
                        else:
                            fp[idx] = 1
                            if show_animation:
                                status = "REPEATED MATCH!"
                else:
                    fp[idx] = 1
                    if ovmax > 0:
                        status = "INSUFFICIENT OVERLAP"

                """
                Draw image to show animation
                """
                if show_animation:
                    height, widht = img.shape[:2]
                    white           = (255,255,255)
                    light_blue      = (255,200,100)
                    green           = (0,255,0)
                    light_red       = (30,30,255)
                    margin          = 10
                    # 1nd line
                    v_pos           = int(height - margin - (bottom_border / 2.0))
                    text            = "Image: " + ground_truth_img[0] + " "
                    img, line_width = draw_text_in_image(img, text, (margin, v_pos), white, 0)
                    text            = "Class [" + str(class_index) + "/" + str(n_classes) + "]: " + class_name + " "
                    img, line_width = draw_text_in_image(img, text, (margin + line_width, v_pos), light_blue, line_width)
                    if ovmax != -1:
                        color       = light_red
                        if status   == "INSUFFICIENT OVERLAP":
                            text    = "IoU: {0:.2f}% ".format(ovmax*100) + "< {0:.2f}% ".format(min_overlap*100)
                        else:
                            text    = "IoU: {0:.2f}% ".format(ovmax*100) + ">= {0:.2f}% ".format(min_overlap*100)
                            color   = green
                        img, _ = draw_text_in_image(img, text, (margin + line_width, v_pos), color, line_width)
                    # 2nd line
                    v_pos           += int(bottom_border / 2.0)
                    rank_pos        = str(idx+1)
                    text            = "Detection #rank: " + rank_pos + " confidence: {0:.2f}% ".format(float(detection["confidence"])*100)
                    img, line_width = draw_text_in_image(img, text, (margin, v_pos), white, 0)
                    color           = light_red
                    if status == "MATCH!":
                        color = green
                    text            = "Result: " + status + " "
                    img, line_width = draw_text_in_image(img, text, (margin + line_width, v_pos), color, line_width)

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    if ovmax > 0: 
                        bbgt = [ int(round(float(x))) for x in gt_match["bbox"].split() ]
                        cv2.rectangle(img,(bbgt[0],bbgt[1]),(bbgt[2],bbgt[3]),light_blue,2)
                        cv2.rectangle(img_cumulative,(bbgt[0],bbgt[1]),(bbgt[2],bbgt[3]),light_blue,2)
                        cv2.putText(img_cumulative, class_name, (bbgt[0],bbgt[1] - 5), font, 0.6, light_blue, 1, cv2.LINE_AA)
                    bb = [int(i) for i in bb]
                    cv2.rectangle(img,(bb[0],bb[1]),(bb[2],bb[3]),color,2)
                    cv2.rectangle(img_cumulative,(bb[0],bb[1]),(bb[2],bb[3]),color,2)
                    cv2.putText(img_cumulative, class_name, (bb[0],bb[1] - 5), font, 0.6, color, 1, cv2.LINE_AA)

                    cv2.imshow("Animation", img)
                    cv2.waitKey(20) 
                    output_img_path = RESULTS_FILES_PATH + "/images/detections_one_by_one/" + class_name + "_detection" + str(idx) + ".jpg"
                    cv2.imwrite(output_img_path, img)
                    cv2.imwrite(img_cumulative_path, img_cumulative)

            cumsum = 0
            for idx, val in enumerate(fp):
                fp[idx] += cumsum
                cumsum += val
                
            cumsum = 0
            for idx, val in enumerate(tp):
                tp[idx] += cumsum
                cumsum += val

            rec = tp[:]
            for idx, val in enumerate(tp):
                rec[idx] = float(tp[idx]) / np.maximum(gt_counter_per_class[class_name], 1)

            prec = tp[:]
            for idx, val in enumerate(tp):
                prec[idx] = float(tp[idx]) / np.maximum((fp[idx] + tp[idx]), 1)

            ap, mrec, mprec = voc_ap(rec[:], prec[:])
            F1  = np.array(rec)*np.array(prec)*2 / np.where((np.array(prec)+np.array(rec))==0, 1, (np.array(prec)+np.array(rec)))

            sum_AP  += ap
            text    = "{0:.2f}%".format(ap*100) + " = " + class_name + " AP " #class_name + " AP = {0:.2f}%".format(ap*100)

            if len(prec)>0:
                F1_text         = "{0:.2f}".format(F1[score05_idx]) + " = " + class_name + " F1 "
                Recall_text     = "{0:.2f}%".format(rec[score05_idx]*100) + " = " + class_name + " Recall "
                Precision_text  = "{0:.2f}%".format(prec[score05_idx]*100) + " = " + class_name + " Precision "
            else:
                F1_text         = "0.00" + " = " + class_name + " F1 " 
                Recall_text     = "0.00%" + " = " + class_name + " Recall " 
                Precision_text  = "0.00%" + " = " + class_name + " Precision " 

            rounded_prec    = [ '%.2f' % elem for elem in prec ]
            rounded_rec     = [ '%.2f' % elem for elem in rec ]
            results_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall :" + str(rounded_rec) + "\n\n")
            if len(prec)>0:
                print(text + "\t||\tscore_threhold=0.5 : " + "F1=" + "{0:.2f}".format(F1[score05_idx])\
                    + " ; Recall=" + "{0:.2f}%".format(rec[score05_idx]*100) + " ; Precision=" + "{0:.2f}%".format(prec[score05_idx]*100))
            else:
                print(text + "\t||\tscore_threhold=0.5 : F1=0.00% ; Recall=0.00% ; Precision=0.00%")
            ap_dictionary[class_name] = ap

            n_images = counter_images_per_class[class_name]
            lamr, mr, fppi = log_average_miss_rate(np.array(rec), np.array(fp), n_images)
            lamr_dictionary[class_name] = lamr

            if draw_plot:
                plt.plot(rec, prec, '-o')
                area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
                area_under_curve_y = mprec[:-1] + [0.0] + [mprec[-1]]
                plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')

                fig = plt.gcf()
                fig.canvas.set_window_title('AP ' + class_name)

                plt.title('class: ' + text)
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                axes = plt.gca()
                axes.set_xlim([0.0,1.0])
                axes.set_ylim([0.0,1.05]) 
                fig.savefig(RESULTS_FILES_PATH + "/AP/" + class_name + ".png")
                plt.cla()

                plt.plot(score, F1, "-", color='orangered')
                plt.title('class: ' + F1_text + "\nscore_threhold=0.5")
                plt.xlabel('Score_Threhold')
                plt.ylabel('F1')
                axes = plt.gca()
                axes.set_xlim([0.0,1.0])
                axes.set_ylim([0.0,1.05])
                fig.savefig(RESULTS_FILES_PATH + "/F1/" + class_name + ".png")
                plt.cla()

                plt.plot(score, rec, "-H", color='gold')
                plt.title('class: ' + Recall_text + "\nscore_threhold=0.5")
                plt.xlabel('Score_Threhold')
                plt.ylabel('Recall')
                axes = plt.gca()
                axes.set_xlim([0.0,1.0])
                axes.set_ylim([0.0,1.05])
                fig.savefig(RESULTS_FILES_PATH + "/Recall/" + class_name + ".png")
                plt.cla()

                plt.plot(score, prec, "-s", color='palevioletred')
                plt.title('class: ' + Precision_text + "\nscore_threhold=0.5")
                plt.xlabel('Score_Threhold')
                plt.ylabel('Precision')
                axes = plt.gca()
                axes.set_xlim([0.0,1.0])
                axes.set_ylim([0.0,1.05])
                fig.savefig(RESULTS_FILES_PATH + "/Precision/" + class_name + ".png")
                plt.cla()
                
        if show_animation:
            cv2.destroyAllWindows()

        results_file.write("\n# mAP of all classes\n")
        mAP     = sum_AP / n_classes
        text    = "mAP = {0:.2f}%".format(mAP*100)
        results_file.write(text + "\n")
        print(text)

    shutil.rmtree(TEMP_FILES_PATH)

    """
    Count total of detection-results
    """
    det_counter_per_class = {}
    for txt_file in dr_files_list:
        lines_list = file_lines_to_list(txt_file)
        for line in lines_list:
            class_name = line.split()[0]
            if class_name in det_counter_per_class:
                det_counter_per_class[class_name] += 1
            else:
                det_counter_per_class[class_name] = 1
    dr_classes = list(det_counter_per_class.keys())

    """
    Write number of ground-truth objects per class to results.txt
    """
    with open(RESULTS_FILES_PATH + "/results.txt", 'a') as results_file:
        results_file.write("\n# Number of ground-truth objects per class\n")
        for class_name in sorted(gt_counter_per_class):
            results_file.write(class_name + ": " + str(gt_counter_per_class[class_name]) + "\n")

    """
    Finish counting true positives
    """
    for class_name in dr_classes:
        if class_name not in gt_classes:
            count_true_positives[class_name] = 0

    """
    Write number of detected objects per class to results.txt
    """
    with open(RESULTS_FILES_PATH + "/results.txt", 'a') as results_file:
        results_file.write("\n# Number of detected objects per class\n")
        for class_name in sorted(dr_classes):
            n_det = det_counter_per_class[class_name]
            text = class_name + ": " + str(n_det)
            text += " (tp:" + str(count_true_positives[class_name]) + ""
            text += ", fp:" + str(n_det - count_true_positives[class_name]) + ")\n"
            results_file.write(text)

    """
    Plot the total number of occurences of each class in the ground-truth
    """
    if draw_plot:
        window_title = "ground-truth-info"
        plot_title = "ground-truth\n"
        plot_title += "(" + str(len(ground_truth_files_list)) + " files and " + str(n_classes) + " classes)"
        x_label = "Number of objects per class"
        output_path = RESULTS_FILES_PATH + "/ground-truth-info.png"
        to_show = False
        plot_color = 'forestgreen'
        draw_plot_func(
            gt_counter_per_class,
            n_classes,
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            '',
            )

    # """
    # Plot the total number of occurences of each class in the "detection-results" folder
    # """
    # if draw_plot:
    #     window_title = "detection-results-info"
    #     # Plot title
    #     plot_title = "detection-results\n"
    #     plot_title += "(" + str(len(dr_files_list)) + " files and "
    #     count_non_zero_values_in_dictionary = sum(int(x) > 0 for x in list(det_counter_per_class.values()))
    #     plot_title += str(count_non_zero_values_in_dictionary) + " detected classes)"
    #     # end Plot title
    #     x_label = "Number of objects per class"
    #     output_path = RESULTS_FILES_PATH + "/detection-results-info.png"
    #     to_show = False
    #     plot_color = 'forestgreen'
    #     true_p_bar = count_true_positives
    #     draw_plot_func(
    #         det_counter_per_class,
    #         len(det_counter_per_class),
    #         window_title,
    #         plot_title,
    #         x_label,
    #         output_path,
    #         to_show,
    #         plot_color,
    #         true_p_bar
    #         )

    """
    Draw log-average miss rate plot (Show lamr of all classes in decreasing order)
    """
    if draw_plot:
        window_title = "lamr"
        plot_title = "log-average miss rate"
        x_label = "log-average miss rate"
        output_path = RESULTS_FILES_PATH + "/lamr.png"
        to_show = False
        plot_color = 'royalblue'
        draw_plot_func(
            lamr_dictionary,
            n_classes,
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            ""
            )

    """
    Draw mAP plot (Show AP's of all classes in decreasing order)
    """
    if draw_plot:
        window_title = "mAP"
        plot_title = "mAP = {0:.2f}%".format(mAP*100)
        x_label = "Average Precision"
        output_path = RESULTS_FILES_PATH + "/mAP.png"
        to_show = True
        plot_color = 'royalblue'
        draw_plot_func(
            ap_dictionary,
            n_classes,
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            ""
            )

def preprocess_gt(gt_path, class_names):
    image_ids   = os.listdir(gt_path)
    results = {}

    images = []
    bboxes = []
    for i, image_id in enumerate(image_ids):
        lines_list      = file_lines_to_list(os.path.join(gt_path, image_id))
        boxes_per_image = []
        image           = {}
        image_id        = os.path.splitext(image_id)[0]
        image['file_name'] = image_id + '.jpg'
        image['width']     = 1
        image['height']    = 1
        #-----------------------------------------------------------------#
        #   感谢 多学学英语吧 的提醒
        #   解决了'Results do not correspond to current coco set'问题
        #-----------------------------------------------------------------#
        image['id']        = int(image_id)

        for line in lines_list:
            difficult = 0 
            if "difficult" in line:
                line_split  = line.split()
                left, top, right, bottom, _difficult = line_split[-5:]
                class_name  = ""
                for name in line_split[:-5]:
                    class_name += name + " "
                class_name  = class_name[:-1]
                difficult = 1
            else:
                line_split  = line.split()
                left, top, right, bottom = line_split[-4:]
                class_name  = ""
                for name in line_split[:-4]:
                    class_name += name + " "
                class_name = class_name[:-1]
            
            left, top, right, bottom = float(left), float(top), float(right), float(bottom)
            cls_id  = class_names.index(class_name) + 1
            bbox    = [left, top, right - left, bottom - top, difficult, int(image_id), cls_id, (right - left) * (bottom - top) - 10.0]
            boxes_per_image.append(bbox)
        images.append(image)
        bboxes.extend(boxes_per_image)
    results['images']        = images

    categories = []
    for i, cls in enumerate(class_names):
        category = {}
        category['supercategory']   = cls
        category['name']            = cls
        category['id']              = i
        categories.append(category)
    results['categories']   = categories

    annotations = []
    for i, box in enumerate(bboxes):
        annotation = {}
        annotation['area']        = box[-1]
        annotation['category_id'] = box[-2]
        annotation['image_id']    = box[-3]
        annotation['iscrowd']     = box[-4]
        annotation['bbox']        = box[:4]
        annotation['id']          = i
        annotations.append(annotation)
    results['annotations'] = annotations
    return results

def preprocess_dr(dr_path, class_names):
    image_ids = os.listdir(dr_path)
    results = []
    for image_id in image_ids:
        lines_list      = file_lines_to_list(os.path.join(dr_path, image_id))
        image_id        = os.path.splitext(image_id)[0]
        for line in lines_list:
            line_split  = line.split()
            confidence, left, top, right, bottom = line_split[-5:]
            class_name  = ""
            for name in line_split[:-5]:
                class_name += name + " "
            class_name  = class_name[:-1]
            left, top, right, bottom = float(left), float(top), float(right), float(bottom)
            result                  = {}
            result["image_id"]      = int(image_id)
            result["category_id"]   = class_names.index(class_name) + 1
            result["bbox"]          = [left, top, right - left, bottom - top]
            result["score"]         = float(confidence)
            results.append(result)
    return results
 
def get_coco_map(class_names, path):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    
    GT_PATH     = os.path.join(path, 'ground-truth')
    DR_PATH     = os.path.join(path, 'detection-results')
    COCO_PATH   = os.path.join(path, 'coco_eval')

    if not os.path.exists(COCO_PATH):
        os.makedirs(COCO_PATH)

    GT_JSON_PATH = os.path.join(COCO_PATH, 'instances_gt.json')
    DR_JSON_PATH = os.path.join(COCO_PATH, 'instances_dr.json')

    with open(GT_JSON_PATH, "w") as f:
        results_gt  = preprocess_gt(GT_PATH, class_names)
        json.dump(results_gt, f, indent=4)

    with open(DR_JSON_PATH, "w") as f:
        results_dr  = preprocess_dr(DR_PATH, class_names)
        json.dump(results_dr, f, indent=4)

    cocoGt      = COCO(GT_JSON_PATH)
    cocoDt      = cocoGt.loadRes(DR_JSON_PATH)
    cocoEval    = COCOeval(cocoGt, cocoDt, 'bbox') 
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
