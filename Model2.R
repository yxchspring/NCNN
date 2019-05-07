#################### Action recognition for still image using VGG16 ##################
#'
#' 1) NCNN:Model2 model
#' 
setwd("Your current path")
### load the requireed package
library(keras)
library(caret) # for confusionMatrix calculation

######################################################################################
#################### NCNN(Model2 model):configure the model
######################################################################################
#############
############# Step 1: Data preparation
#############
############# Step 1.1: Configure the train, val, and test folders
classiftime1 <- proc.time()  # record classification time 
base_dir <- "./Willow-actions/cropped"
train_dir <- file.path(base_dir, "train")
val_dir <- file.path(base_dir, "val")
test_dir <- file.path(base_dir, "test")

############# Step 1.2: Configure the model parameters for the pre-trained vgg16 model
### a) Load the Baseline_weights.1-0.66.hdf5
ckp_path <- "./Results/Willow-actions/cropped/checkpoints"
model_baseline_best <- load_model_hdf5(
  file.path(ckp_path, "Baseline_weights.19-0.66.hdf5")
)
summary(model_baseline_best)

### b) Conduct data augmentation for training set and 
###    set the the data generator for validation and testing sets
# For training set
datagen_train <- image_data_generator(
  samplewise_center = TRUE,
  # rescale = 1/255,
  rotation_range = 90,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE,
  vertical_flip = TRUE,
  fill_mode = "nearest"
)

# For validation and testing sets
datagen_val_test <- image_data_generator(
  samplewise_center = TRUE
  # samplewise_std_normalization=TRUE
  # zca_whitening = TRUE,
  # rescale = 1/255
)

batch_size <- 20

### c) Configure the flow for reading images (It will benefit from the batch processing)
# For training set 
train_generator <- flow_images_from_directory(
  train_dir,
  datagen_train,
  target_size = c(224, 224),
  batch_size = batch_size,
  class_mode = "categorical"
)
# For validation set 
val_generator <- flow_images_from_directory(
  val_dir,
  datagen_val_test,
  target_size = c(224, 224),
  batch_size = batch_size,
  class_mode = "categorical"
)
# For testing set 
test_generator <- flow_images_from_directory(
  test_dir,
  datagen_val_test,
  target_size = c(224, 224),
  batch_size = batch_size,
  class_mode = "categorical",
  # This is very importance, and it benfits to acquire the true labels
  shuffle = FALSE  # make sure the same 
)

############# Code comment start!!!
# #############
# ############# Step 2: Build the model
# #############
# ############# Step 2.1: Configure the input and output layers of softmax classifier
# 
# ### a) Model preparation
# # Get all the layers of baseline models
# layers_baseline <- model_baseline_best$layers
# 
# # Show the names of layers
# # for (i in 1:length(layers_baseline)) {
# #   cat(i,layers_baseline[[i]]$name,"\n")
# # }
# 
# # Assign the specific names to the layers
# names(layers_baseline) <- c('input_1', 
#                             'block1_conv1', 'block1_conv2', 'block1_pool', 
#                             'block2_conv1', 'block2_conv2', 'block2_pool', 
#                             'block3_conv1', 'block3_conv2', 'block3_conv3', 'block3_pool', 
#                             'block4_conv1', 'block4_conv2', 'block4_conv3', 'block4_pool', 
#                             'block5_conv1', 'block5_conv2', 'block5_conv3', 'block5_pool',
#                             'GAP', 'dense_1','dense_2', 'dense_3')
# 
# ### b) Configure the input layer for softmax classifier
# # Configure the input of keras model
# keras_input <- layers_baseline$block5_pool$output
# 
# ### c) Configure the separable convolution layers
# # Configure each branch
# branch_a <- keras_input %>%
#   layer_conv_2d(filters = 128, kernel_size = 1, padding = "same",
#                 activation = "relu", strides = 1)
# 
# branch_b <- keras_input %>%
#   layer_conv_2d(filters = 128, kernel_size = 1, padding = "same",
#                 activation = "relu") %>%
#   layer_conv_2d(filters = 128, kernel_size = 3, padding = "same",
#                 activation = "relu", strides = 1)
# 
# branch_c <- keras_input %>%
#   layer_average_pooling_2d(pool_size = 3, strides = 1, padding = "same") %>%
#   layer_conv_2d(filters = 128, kernel_size = 3, padding = "same",
#                 activation = "relu")
# 
# # Concatenate each branch
# output_concatenate <- layer_concatenate(list(branch_a, branch_b, branch_c),
#                                         axis = -1)
# 
# # Configure softmax layers
# keras_output_Baseline <- output_concatenate %>%
#   # layer_flatten() %>%
#   layer_global_average_pooling_2d()%>%
#   layer_dense(units = 2048, activation = 'relu',kernel_regularizer = regularizer_l2(0.001)) %>% 
#   # layer_dropout(0.5) %>%
#   layer_dense(units = 2048, activation = 'relu',kernel_regularizer = regularizer_l2(0.001)) %>%
#   # layer_dropout(0.5) %>%
#   layer_dense(units = 7, activation = 'sigmoid')
# 
# ############# Step 2.2: Integrate the final keras model of baseline
# ### a) Determine the input and output for the whole CNN network
# model_Model2 <- keras_model(inputs = model_baseline_best$input,outputs = keras_output_Baseline)
# summary(model_Model2)
# 
# ### b) Output the names of layers for the whole CNN network
# for (i in 1:length(model_Model2$layers)) {
#   cat(i,model_Model2$layers[[i]]$name,"\n")
# }
# 
# ########################################################################################################
# ### c) Set the weights for the model_Model2 using the previously trained model:model_Baseline_best
# ########################################################################################################
# # set the weights for the following layers:
# # 'GAP', 'dense_1','dense_2', 'dense_3'
# 
# # 1) for 'dense_1'
# # set_weights(model_Model2$layers[[27]],get_weights(layers_baseline$dense_1))
# 
# # 2) for 'dense_2'
# set_weights(model_Model2$layers[[28]],get_weights(layers_baseline$dense_2))
# 
# # 3) for 'dense_3'
# set_weights(model_Model2$layers[[29]],get_weights(layers_baseline$dense_3))
# 
# ### c) Freeze the pre-trained bottom layers
# # Make sure the pre-trained bottom layers are not trainable
# for (layer in model_Model2$layers[c(1:19)]) {
#   layer$trainable <- FALSE
# }
# 
# ############# Step 2.3: Create the checkpoint callback
# ckp_path <- "./Results/Willow-actions/cropped/checkpoints"
# ckp_filename_path <- file.path(ckp_path, "Model2_weights.{epoch:02d}-{val_acc:.2f}.hdf5")
# 
# ckp_callback <- callback_model_checkpoint(
#   filepath = ckp_filename_path,
#   monitor = "val_acc",
#   # monitor = "val_loss",
#   save_weights_only = FALSE,
#   save_best_only = TRUE,
#   verbose = 1
# )
# 
# ############# Step 2.4: Traning the model_Model2
# model_Model2 %>% compile(
#   loss = "categorical_crossentropy",
#   optimizer = optimizer_rmsprop(lr = 1e-4),
#   metrics = c("accuracy")
# )
# history_fit <- model_Model2 %>% fit_generator(
#   train_generator,
#   steps_per_epoch = ceiling(train_generator$samples/batch_size), # 30*6/10=18 
#   epochs = 50, # 30
#   validation_data = val_generator,
#   validation_steps = ceiling(val_generator$samples/batch_size), # 10*6/10=6
#   callbacks = list(ckp_callback)  # pass callback to training
# )
# classiftime <-  proc.time() - classiftime1 
# cat("The CPUTime is: ",classiftime," \n")
# cat("This is the number of trainable weights after freezing",
#     "model_Model2:", length(model_Model2$trainable_weights), "\n")
# 
# layer_list_final <- model_Model2$layers
# ############# Step 2.5: Check and Save the acc and loss curves for train+val
# ### a) Check the saved model weight
# list.files(ckp_path)
# 
# ### b) Save the figure of acc and loss curves for train+val
# fig_path <- "./Results/Willow-actions/cropped/figures"
# fig_train_val <- paste0(fig_path,"/Model2_fig.png")
# # png(filename = file_subtype_ROC,width = 5,height = 5,res = 300)
# plot(history_fit,colorize=TRUE, main="Loss and Acc Curves for traning and validation set",
#      cex=1.5, cex.main=1.2, cex.lab=1.5, cex.axis=1.5,cex.sub=1.5)
# dev.copy(png, fig_train_val, width=10, height=6,res=300,units="in")
# dev.off()
############# Code comment ends!!!

#############
############# Step 4: Evaluate your model
#############
############# Step 4.1: Load the best trained model
# Load the weights from the latest checkpoint (epoch 50), and re-evaluate:
######################## Top1 best model ###########################
ckp_path <- "./Results/Willow-actions/cropped/checkpoints"
model_Model2_best <- load_model_hdf5(
  file.path(ckp_path, "Model2_weights.49-0.68.hdf5")
)

######################## Top2 best model ###########################
# model_Model2_best <- load_model_hdf5(
#   file.path(ckp_path, "Model2_weights.47-0.66.hdf5")
# )

############# Step 4.2: Re-evaluate the model on testing set
### a) Get the acc and loss
result_Model2 <- model_Model2_best %>% evaluate_generator(test_generator,
                                                          steps = ceiling(test_generator$samples/batch_size)) # 20*6/10=12
# result_Model2

### b) Obatin the prediction probability
result_pred_Model2 <- model_Model2_best %>% predict_generator(test_generator,
                                                              steps = ceiling(test_generator$samples/batch_size)) # 20*6/10=12
# result_pred_Model2

### c) Obatin the prediction classes
result_pred_class_Model2 <- apply(result_pred_Model2, 1, function(x) {which.max(x)-1})  # 20*6/10=12
# result_pred_class_Model2

cat("The task for re-evaluating the model on testing set has been competed!\n")

#############
############# Step 5: Summaryize the final model
#############
### a) Get the class index for each class
class_indices_test <- test_generator$class_indices

### b) Get the true labels of test set
# The class index of each sampe is based on the sorted order of folder names
true_labels_test <- test_generator$classes

### c) Get the confusionmatrix
cfn_matrix_list <- confusionMatrix(factor(result_pred_class_Model2),factor(true_labels_test))

cfn_matrix <- as.matrix(cfn_matrix_list)
cfn_matrix_prob <- sweep(cfn_matrix,2,table(true_labels_test),"/")
cfn_matrix_overall <- as.matrix(cfn_matrix_list, what = "overall")
cfn_matrix_classes <- as.matrix(cfn_matrix_list, what = "classes")

Results_data_filename <- './Results/Willow-actions/cropped/data/Model2_results.Rdata'
save(result_Model2,class_indices_test,cfn_matrix,cfn_matrix_prob,cfn_matrix_overall,cfn_matrix_classes,
     file = Results_data_filename)
cat("The task for data saving for Willow-actions/cropped set has been competed!\n")

### d) Plot the matrix using method 2
require(reshape2)
cfn_matrix_prob <- round(cfn_matrix_prob,digits = 2) # Formating Decimal places
cfn_matrix_prob <- round(cfn_matrix_prob,digits = 2) # Formating Decimal places
rownames(cfn_matrix_prob) <- c('InteractingWithComputer','Photographing','PlayingMusic','RidingBike','RidingHorse','Running','Walking')
colnames(cfn_matrix_prob) <- c('InteractingWithComputer','Photographing','PlayingMusic','RidingBike','RidingHorse','Running','Walking')
reoder_index_dec <- order(rownames(cfn_matrix_prob),decreasing = TRUE)
reoder_index_inc <- order(rownames(cfn_matrix_prob),decreasing = FALSE)
cfn_matrix_prob <-  cfn_matrix_prob[reoder_index_inc, reoder_index_dec]
melted_cfn_matrix_prob <- melt(cfn_matrix_prob)
colnames(melted_cfn_matrix_prob) <- c("class_x","class_y","acc")
require(ggplot2)
classification_heatmap <- ggplot(data = melted_cfn_matrix_prob, aes(x=class_x,y=class_y,fill=acc))+
  geom_tile(color = "white")+
  scale_fill_gradient2(low = "gray", high = "red", mid = "white", 
                       midpoint = 0.5, limit = c(0,1), space = "Lab", 
                       name="Acc") +
  theme_minimal()+ # minimal theme
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 18, hjust = 1),
        axis.text.y = element_text(angle = 0, vjust = 1, 
                                   size = 18, hjust = 1))+
  
  coord_fixed() +
  geom_text(aes(class_x, class_y, label = acc), color = "black", size = 6) +
  theme(
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    panel.grid.major = element_blank(),
    panel.border = element_blank(),
    panel.background = element_blank(),
    axis.ticks = element_blank(),
    # legend.justification = c(0, 1),
    # legend.position = c(1, 0.7),
    legend.direction = "horizontal")+
  guides(fill = guide_colorbar(barwidth = 1, barheight =7,direction = "vertical",
                               title.position = "top", title.hjust = 0.5))
# Print the heatmap
print(classification_heatmap)

# Save the ggplot
cfnMatrix_filename <- './Results/Willow-actions/cropped/figures/Model2_cMatrix_ggplot.png'
ggsave(cfnMatrix_filename,width=9, height=9,dpi = 300,units="in")
cat("The task to save confusionMatirx ggplot for Willow-actions/cropped set has been competed!\n")

cat("The task for action recogniton on still image for Willow-actions/cropped set has been competed!\n")






