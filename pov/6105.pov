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
    sphere { m*<-1.5270932676382616,-0.29753192547220164,-0.9926269273077871>, 1 }        
    sphere {  m*<-0.0644060400916433,0.19120359915356855,8.887829510191473>, 1 }
    sphere {  m*<7.290945397908334,0.10228332315921135,-5.691663779853888>, 1 }
    sphere {  m*<-3.653263203852215,2.5822874032565846,-2.0829242637473087>, 1}
    sphere { m*<-2.9066938947647856,-2.872836883571173,-1.672927844935483>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.0644060400916433,0.19120359915356855,8.887829510191473>, <-1.5270932676382616,-0.29753192547220164,-0.9926269273077871>, 0.5 }
    cylinder { m*<7.290945397908334,0.10228332315921135,-5.691663779853888>, <-1.5270932676382616,-0.29753192547220164,-0.9926269273077871>, 0.5}
    cylinder { m*<-3.653263203852215,2.5822874032565846,-2.0829242637473087>, <-1.5270932676382616,-0.29753192547220164,-0.9926269273077871>, 0.5 }
    cylinder {  m*<-2.9066938947647856,-2.872836883571173,-1.672927844935483>, <-1.5270932676382616,-0.29753192547220164,-0.9926269273077871>, 0.5}

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
    sphere { m*<-1.5270932676382616,-0.29753192547220164,-0.9926269273077871>, 1 }        
    sphere {  m*<-0.0644060400916433,0.19120359915356855,8.887829510191473>, 1 }
    sphere {  m*<7.290945397908334,0.10228332315921135,-5.691663779853888>, 1 }
    sphere {  m*<-3.653263203852215,2.5822874032565846,-2.0829242637473087>, 1}
    sphere { m*<-2.9066938947647856,-2.872836883571173,-1.672927844935483>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.0644060400916433,0.19120359915356855,8.887829510191473>, <-1.5270932676382616,-0.29753192547220164,-0.9926269273077871>, 0.5 }
    cylinder { m*<7.290945397908334,0.10228332315921135,-5.691663779853888>, <-1.5270932676382616,-0.29753192547220164,-0.9926269273077871>, 0.5}
    cylinder { m*<-3.653263203852215,2.5822874032565846,-2.0829242637473087>, <-1.5270932676382616,-0.29753192547220164,-0.9926269273077871>, 0.5 }
    cylinder {  m*<-2.9066938947647856,-2.872836883571173,-1.672927844935483>, <-1.5270932676382616,-0.29753192547220164,-0.9926269273077871>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    