#version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<-0.8964424822824709,-1.166341277033018,-0.6691406604457311>, 1 }        
    sphere {  m*<0.5304791099433113,-0.30431654089862537,9.190990085776262>, 1 }
    sphere {  m*<7.885830547943281,-0.3932368168929816,-5.388503204269071>, 1 }
    sphere {  m*<-6.585378207264105,5.590223224172407,-3.5806213022800963>, 1}
    sphere { m*<-2.123864417370899,-3.8362178447891213,-1.2720862008196179>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5304791099433113,-0.30431654089862537,9.190990085776262>, <-0.8964424822824709,-1.166341277033018,-0.6691406604457311>, 0.5 }
    cylinder { m*<7.885830547943281,-0.3932368168929816,-5.388503204269071>, <-0.8964424822824709,-1.166341277033018,-0.6691406604457311>, 0.5}
    cylinder { m*<-6.585378207264105,5.590223224172407,-3.5806213022800963>, <-0.8964424822824709,-1.166341277033018,-0.6691406604457311>, 0.5 }
    cylinder {  m*<-2.123864417370899,-3.8362178447891213,-1.2720862008196179>, <-0.8964424822824709,-1.166341277033018,-0.6691406604457311>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    #version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<-0.8964424822824709,-1.166341277033018,-0.6691406604457311>, 1 }        
    sphere {  m*<0.5304791099433113,-0.30431654089862537,9.190990085776262>, 1 }
    sphere {  m*<7.885830547943281,-0.3932368168929816,-5.388503204269071>, 1 }
    sphere {  m*<-6.585378207264105,5.590223224172407,-3.5806213022800963>, 1}
    sphere { m*<-2.123864417370899,-3.8362178447891213,-1.2720862008196179>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5304791099433113,-0.30431654089862537,9.190990085776262>, <-0.8964424822824709,-1.166341277033018,-0.6691406604457311>, 0.5 }
    cylinder { m*<7.885830547943281,-0.3932368168929816,-5.388503204269071>, <-0.8964424822824709,-1.166341277033018,-0.6691406604457311>, 0.5}
    cylinder { m*<-6.585378207264105,5.590223224172407,-3.5806213022800963>, <-0.8964424822824709,-1.166341277033018,-0.6691406604457311>, 0.5 }
    cylinder {  m*<-2.123864417370899,-3.8362178447891213,-1.2720862008196179>, <-0.8964424822824709,-1.166341277033018,-0.6691406604457311>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    