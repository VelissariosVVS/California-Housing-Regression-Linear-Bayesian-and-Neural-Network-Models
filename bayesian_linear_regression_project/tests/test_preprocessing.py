from preprocessing import prepare_data

data = prepare_data()

print(data["X_train_scaled"].shape)
print(data["X_test_scaled"].shape)
print(data["y_train"].shape)
print(data["y_test"].shape)