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
    sphere { m*<-1.274012555023917e-18,1.2726258373002853e-18,0.06593586898869976>, 1 }        
    sphere {  m*<-3.2639906061925317e-18,-8.663891956573447e-19,9.708935868988696>, 1 }
    sphere {  m*<9.428090415820634,-3.4092373467770986e-19,-3.2673974643446346>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-3.2673974643446346>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-3.2673974643446346>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-3.2639906061925317e-18,-8.663891956573447e-19,9.708935868988696>, <-1.274012555023917e-18,1.2726258373002853e-18,0.06593586898869976>, 0.5 }
    cylinder { m*<9.428090415820634,-3.4092373467770986e-19,-3.2673974643446346>, <-1.274012555023917e-18,1.2726258373002853e-18,0.06593586898869976>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-3.2673974643446346>, <-1.274012555023917e-18,1.2726258373002853e-18,0.06593586898869976>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-3.2673974643446346>, <-1.274012555023917e-18,1.2726258373002853e-18,0.06593586898869976>, 0.5}

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
    sphere { m*<-1.274012555023917e-18,1.2726258373002853e-18,0.06593586898869976>, 1 }        
    sphere {  m*<-3.2639906061925317e-18,-8.663891956573447e-19,9.708935868988696>, 1 }
    sphere {  m*<9.428090415820634,-3.4092373467770986e-19,-3.2673974643446346>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-3.2673974643446346>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-3.2673974643446346>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-3.2639906061925317e-18,-8.663891956573447e-19,9.708935868988696>, <-1.274012555023917e-18,1.2726258373002853e-18,0.06593586898869976>, 0.5 }
    cylinder { m*<9.428090415820634,-3.4092373467770986e-19,-3.2673974643446346>, <-1.274012555023917e-18,1.2726258373002853e-18,0.06593586898869976>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-3.2673974643446346>, <-1.274012555023917e-18,1.2726258373002853e-18,0.06593586898869976>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-3.2673974643446346>, <-1.274012555023917e-18,1.2726258373002853e-18,0.06593586898869976>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    