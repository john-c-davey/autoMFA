##' @export
summary.MFA <- function(object, ...){
  print(object$diagnostics$call)
  print(data.frame("No components" = length(object$model$pivec), "log_like" = object$diagnostics$logL, "BIC" = object$diagnostics$bic))
  cat("Component specific numbers of factors:", "\n",object$model$numFactors, "\n" )
}

##' @export
plot.MFA <- function(x, ...){
  my_col <- ggsci::pal_jco()(ncol(x$diagnostics$data))
  if(is.null(colnames(x$diagnostics$data))){
    v_labels = paste("V",1:ncol(x$diagnostics$data),sep="")
  }else{
    v_labels = colnames(x$diagnostics$data)
  }
  graphics::pairs(x$diagnostics$data, lower.panel = NULL, col = my_col[x$clustering$allocations], labels = v_labels)
}

##' @export
print.MFA <- function(x, ...){
  print(x$diagnostics$call)
  cat("The mixing proportions are:", "\n")
  print(x$model$pivec)
  cat("The component means are:", "\n")
  print(x$model$mu)
  cat("The factor loading matrices are:", "\n")
  print(x$model$B)
  cat("The error variance matrices are:", "\n")
  print(x$model$D)  
}
