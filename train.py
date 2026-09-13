from ultralytics import YOLO


def main():
    model = YOLO("yolo26n.pt")

    model.train(
        data=r"C:\Users\李佳钰\PycharmProjects\stair_obstacle_project\dataset_800\data.yaml",
        epochs=60,
        imgsz=640,
        batch=4,
        device=0,
        workers=0,
        project=r"C:\Users\李佳钰\PycharmProjects\stair_obstacle_project\runs",
        name="trash_overflow_800",
        patience=15,
        cache=False
    )


if __name__ == "__main__":
    main()