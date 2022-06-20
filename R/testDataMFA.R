#' Test dataset for the MFA model
#'
#' A 720 x 3 test dataset generated from a MFA model with 3 components, 1 factor for 
#' each component. Uneven point distribution with large separation between 
#' clusters relative to the component variance matrices. 
#'
#' @docType data
#' @usage testDataMFA
#'
#' @format Data matrix with 720 observations of 3 variables. Generated using an MFA model with the following parameters:
#' \itemize{
#' \item{\code{pivec}}{ Mixing proportion vector (0.5722, 0.3333, 0.0944) which corresponds to component sizes of 412, 240 and 68.}
#' \item{\code{mu}}{ Mean vectors (3;0;0), (0;3;0) and (0,0,3) respectively.}
#' \item{\code{B}}{ Loading matrices (0.8827434; -0.5617922; 0.0277005), (0.03121194; 0.14964642; 0.01180723) and (0.1306169; 0.7450665; 0.4357088) respectively.}
#' \item{\code{D}}{ Error variance matrices of diag(0.1) for all components.}
#' }
#' 
#' 
#'
#' @keywords datasets
#' @examples
#' pairs(testDataMFA) 
"testDataMFA"