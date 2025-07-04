# Библиотеки
library(tidyverse)
library(glmnet)
library(caret)
library(pROC)
library(ggplot2)

# Загрузка и первичная очистка
cancer_data <- read.csv("data.csv", stringsAsFactors = FALSE)

# Удаляем неинформативные столбцы
cancer_data <- cancer_data %>% select(-id, -X)  # 'X' — это, скорее всего, пустой столбец

# Преобразуем diagnosis в бинарную переменную (фактор с валидными уровнями)
cancer_data$diagnosis_bin <- factor(
  ifelse(cancer_data$diagnosis == "M", 1, 0),
  levels = c(0, 1),
  labels = c("Benign", "Malignant")
)

# Проверка на NA
if (sum(is.na(cancer_data)) > 0) {
  cat("Обнаружены NA в данных. Рекомендуется заполнить их, например, средними значениями.\n")
  # Пример заполнения NA средними значениями (раскомментируйте, если нужно)
  # cancer_data <- cancer_data %>% mutate(across(where(is.numeric), ~replace(., is.na(.), mean(., na.rm = TRUE))))
}

# Деление на train/test
set.seed(123)
train_idx <- createDataPartition(cancer_data$diagnosis_bin, p = 0.8, list = FALSE)
train_data <- cancer_data[train_idx, ]
test_data  <- cancer_data[-train_idx, ]

# Очистка NA и синхронизация данных
train_data_clean <- train_data %>% drop_na()
test_data_clean  <- test_data %>% drop_na()

# Извлечение признаков и целевой переменной
features_train <- train_data_clean %>% select(-diagnosis, -diagnosis_bin)
features_test  <- test_data_clean %>% select(-diagnosis, -diagnosis_bin)
y_train <- train_data_clean$diagnosis_bin
y_test  <- test_data_clean$diagnosis_bin

# Приводим все признаки к числовому формату
features_train <- features_train %>% mutate(across(everything(), as.numeric))
features_test  <- features_test %>% mutate(across(everything(), as.numeric))

# Масштабирование признаков (стандартизация)
preproc <- preProcess(features_train, method = c("center", "scale"))
features_train_scaled <- predict(preproc, features_train)
features_test_scaled  <- predict(preproc, features_test)

# Преобразование в матрицы для glmnet
x_train <- as.matrix(features_train_scaled)
x_test  <- as.matrix(features_test_scaled)

# Обучение модели: LASSO (L1)
cv <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 1)
model_lasso <- glmnet(x_train, y_train, family = "binomial", alpha = 1, lambda = cv$lambda.min)

# 10-кратная кросс-валидация для оценки стабильности
ctrl <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)
lasso_cv <- train(
  x = x_train,
  y = y_train,
  method = "glmnet",
  family = "binomial",
  tuneGrid = expand.grid(alpha = 1, lambda = cv$lambda.min),
  trControl = ctrl
)
cat("Результаты 10-кратной кросс-валидации:\n")
print(lasso_cv)

# Предсказания
pred_probs <- predict(model_lasso, newx = x_test, s = cv$lambda.min, type = "response")

# Оптимизация порога с помощью статистики Юдена
roc_obj <- roc(response = y_test, predictor = as.vector(pred_probs))
optimal_threshold <- coords(roc_obj, "best", ret = "threshold", best.method = "youden")$threshold
cat("Оптимальный порог (по Юдену):", round(optimal_threshold, 3), "\n")

# Создание предсказанных меток с оптимальным порогом
pred_labels <- factor(
  ifelse(pred_probs > optimal_threshold, 1, 0),
  levels = c(0, 1),
  labels = c("Benign", "Malignant")
)

# Матрица ошибок
conf_mat <- confusionMatrix(
  pred_labels,
  y_test,
  positive = "Malignant"
)

# Метрики
accuracy  <- conf_mat$overall["Accuracy"]
precision <- conf_mat$byClass["Pos Pred Value"]
recall    <- conf_mat$byClass["Sensitivity"]
f1        <- 2 * (precision * recall) / (precision + recall)

# Вывод метрик
cat("Accuracy:",  round(accuracy, 3), "\n")
cat("Precision:", round(precision, 3), "\n")
cat("Recall:",    round(recall, 3), "\n")
cat("F1-score:",  round(f1, 3), "\n")
cat("ROC AUC:", round(auc(roc_obj), 3), "\n")

# ROC-кривая
roc_df <- data.frame(
  fpr = 1 - roc_obj$specificities,
  tpr = roc_obj$sensitivities
)
roc_plot <- ggplot(roc_df, aes(x = fpr, y = tpr)) +
  geom_line(color = "blue", linewidth = 1) +
  geom_abline(linetype = "dashed", color = "gray") +
  xlab("Ложно-положительная доля (FPR)") +
  ylab("Истинно-положительная доля (TPR)") +
  ggtitle(paste("ROC-кривая (AUC =", round(auc(roc_obj), 3), ")")) +
  theme_minimal()
print(roc_plot)

# Получение коэффициентов из модели
coef_lasso <- coef(model_lasso)
coef_df <- data.frame(
  Feature = rownames(coef_lasso),
  Coefficient = as.vector(coef_lasso)
)

# Удаляем нулевые коэффициенты и перехват
coef_df <- coef_df %>%
  filter(Feature != "(Intercept)" & Coefficient != 0) %>%
  mutate(AbsCoeff = abs(Coefficient))

# Визуализация важности признаков
importance_plot <- ggplot(coef_df, aes(x = reorder(Feature, AbsCoeff), y = Coefficient, fill = Coefficient > 0)) +
  geom_col(show.legend = TRUE) +
  coord_flip() +
  labs(
    x = "Признак",
    y = "Коэффициент (лог-odds)",
    fill = "Знак"
  ) +
  scale_fill_manual(values = c("red", "blue"), labels = c("Отрицательный", "Положительный")) +
  ggtitle("Важность признаков (LASSO, ненулевые коэффициенты)") +
  theme_minimal()
print(importance_plot)


