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
    sphere { m*<-1.266525467109657,-0.694225656573344,-0.8587406379386608>, 1 }        
    sphere {  m*<0.18059826052399286,-0.014422132673119314,9.01269606895807>, 1 }
    sphere {  m*<7.535949698523969,-0.1033424086674759,-5.566797221087278>, 1 }
    sphere {  m*<-4.969842494931152,3.994871173048588,-2.7557993357389945>, 1}
    sphere { m*<-2.569059495067045,-3.3192405136834267,-1.499856007985403>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.18059826052399286,-0.014422132673119314,9.01269606895807>, <-1.266525467109657,-0.694225656573344,-0.8587406379386608>, 0.5 }
    cylinder { m*<7.535949698523969,-0.1033424086674759,-5.566797221087278>, <-1.266525467109657,-0.694225656573344,-0.8587406379386608>, 0.5}
    cylinder { m*<-4.969842494931152,3.994871173048588,-2.7557993357389945>, <-1.266525467109657,-0.694225656573344,-0.8587406379386608>, 0.5 }
    cylinder {  m*<-2.569059495067045,-3.3192405136834267,-1.499856007985403>, <-1.266525467109657,-0.694225656573344,-0.8587406379386608>, 0.5}

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
    sphere { m*<-1.266525467109657,-0.694225656573344,-0.8587406379386608>, 1 }        
    sphere {  m*<0.18059826052399286,-0.014422132673119314,9.01269606895807>, 1 }
    sphere {  m*<7.535949698523969,-0.1033424086674759,-5.566797221087278>, 1 }
    sphere {  m*<-4.969842494931152,3.994871173048588,-2.7557993357389945>, 1}
    sphere { m*<-2.569059495067045,-3.3192405136834267,-1.499856007985403>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.18059826052399286,-0.014422132673119314,9.01269606895807>, <-1.266525467109657,-0.694225656573344,-0.8587406379386608>, 0.5 }
    cylinder { m*<7.535949698523969,-0.1033424086674759,-5.566797221087278>, <-1.266525467109657,-0.694225656573344,-0.8587406379386608>, 0.5}
    cylinder { m*<-4.969842494931152,3.994871173048588,-2.7557993357389945>, <-1.266525467109657,-0.694225656573344,-0.8587406379386608>, 0.5 }
    cylinder {  m*<-2.569059495067045,-3.3192405136834267,-1.499856007985403>, <-1.266525467109657,-0.694225656573344,-0.8587406379386608>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    