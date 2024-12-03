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
    sphere { m*<-1.5691266977000992e-18,-5.208666554484459e-18,1.0828054073787552>, 1 }        
    sphere {  m*<-2.2013360015292216e-18,-4.452750383049923e-18,4.831805407378793>, 1 }
    sphere {  m*<9.428090415820634,3.268495706983578e-19,-2.2505279259545787>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.2505279259545787>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.2505279259545787>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-2.2013360015292216e-18,-4.452750383049923e-18,4.831805407378793>, <-1.5691266977000992e-18,-5.208666554484459e-18,1.0828054073787552>, 0.5 }
    cylinder { m*<9.428090415820634,3.268495706983578e-19,-2.2505279259545787>, <-1.5691266977000992e-18,-5.208666554484459e-18,1.0828054073787552>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.2505279259545787>, <-1.5691266977000992e-18,-5.208666554484459e-18,1.0828054073787552>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.2505279259545787>, <-1.5691266977000992e-18,-5.208666554484459e-18,1.0828054073787552>, 0.5}

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
    sphere { m*<-1.5691266977000992e-18,-5.208666554484459e-18,1.0828054073787552>, 1 }        
    sphere {  m*<-2.2013360015292216e-18,-4.452750383049923e-18,4.831805407378793>, 1 }
    sphere {  m*<9.428090415820634,3.268495706983578e-19,-2.2505279259545787>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.2505279259545787>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.2505279259545787>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-2.2013360015292216e-18,-4.452750383049923e-18,4.831805407378793>, <-1.5691266977000992e-18,-5.208666554484459e-18,1.0828054073787552>, 0.5 }
    cylinder { m*<9.428090415820634,3.268495706983578e-19,-2.2505279259545787>, <-1.5691266977000992e-18,-5.208666554484459e-18,1.0828054073787552>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.2505279259545787>, <-1.5691266977000992e-18,-5.208666554484459e-18,1.0828054073787552>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.2505279259545787>, <-1.5691266977000992e-18,-5.208666554484459e-18,1.0828054073787552>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    