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
    sphere { m*<-2.269568413373463e-18,-5.2011855457205945e-18,1.0269342092033966>, 1 }        
    sphere {  m*<-1.5341857460909509e-18,-4.599342337567677e-18,5.132934209203426>, 1 }
    sphere {  m*<9.428090415820634,9.599008677231134e-20,-2.306399124129936>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.306399124129936>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.306399124129936>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-1.5341857460909509e-18,-4.599342337567677e-18,5.132934209203426>, <-2.269568413373463e-18,-5.2011855457205945e-18,1.0269342092033966>, 0.5 }
    cylinder { m*<9.428090415820634,9.599008677231134e-20,-2.306399124129936>, <-2.269568413373463e-18,-5.2011855457205945e-18,1.0269342092033966>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.306399124129936>, <-2.269568413373463e-18,-5.2011855457205945e-18,1.0269342092033966>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.306399124129936>, <-2.269568413373463e-18,-5.2011855457205945e-18,1.0269342092033966>, 0.5}

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
    sphere { m*<-2.269568413373463e-18,-5.2011855457205945e-18,1.0269342092033966>, 1 }        
    sphere {  m*<-1.5341857460909509e-18,-4.599342337567677e-18,5.132934209203426>, 1 }
    sphere {  m*<9.428090415820634,9.599008677231134e-20,-2.306399124129936>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.306399124129936>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.306399124129936>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-1.5341857460909509e-18,-4.599342337567677e-18,5.132934209203426>, <-2.269568413373463e-18,-5.2011855457205945e-18,1.0269342092033966>, 0.5 }
    cylinder { m*<9.428090415820634,9.599008677231134e-20,-2.306399124129936>, <-2.269568413373463e-18,-5.2011855457205945e-18,1.0269342092033966>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.306399124129936>, <-2.269568413373463e-18,-5.2011855457205945e-18,1.0269342092033966>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.306399124129936>, <-2.269568413373463e-18,-5.2011855457205945e-18,1.0269342092033966>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    