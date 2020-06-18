import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_model_instance_segmentation(num_classes):
    # COCO 에서 미리 학습된 인스턴스 분할 모델을 읽어옵니다
    #model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    
    # 분류를 위한 입력 특징 차원을 얻습니다
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # 미리 학습된 헤더를 새로운 것으로 바꿉니다
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
