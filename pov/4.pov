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
    sphere { m*<-2.963168738146546e-19,8.54931595814368e-19,0.006479733728852102>, 1 }        
    sphere {  m*<8.375233338991941e-20,5.311925216332624e-19,9.971479733728852>, 1 }
    sphere {  m*<9.428090415820634,2.201191479415133e-19,-3.326853599604482>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-3.326853599604482>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-3.326853599604482>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<8.375233338991941e-20,5.311925216332624e-19,9.971479733728852>, <-2.963168738146546e-19,8.54931595814368e-19,0.006479733728852102>, 0.5 }
    cylinder { m*<9.428090415820634,2.201191479415133e-19,-3.326853599604482>, <-2.963168738146546e-19,8.54931595814368e-19,0.006479733728852102>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-3.326853599604482>, <-2.963168738146546e-19,8.54931595814368e-19,0.006479733728852102>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-3.326853599604482>, <-2.963168738146546e-19,8.54931595814368e-19,0.006479733728852102>, 0.5}

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
    sphere { m*<-2.963168738146546e-19,8.54931595814368e-19,0.006479733728852102>, 1 }        
    sphere {  m*<8.375233338991941e-20,5.311925216332624e-19,9.971479733728852>, 1 }
    sphere {  m*<9.428090415820634,2.201191479415133e-19,-3.326853599604482>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-3.326853599604482>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-3.326853599604482>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<8.375233338991941e-20,5.311925216332624e-19,9.971479733728852>, <-2.963168738146546e-19,8.54931595814368e-19,0.006479733728852102>, 0.5 }
    cylinder { m*<9.428090415820634,2.201191479415133e-19,-3.326853599604482>, <-2.963168738146546e-19,8.54931595814368e-19,0.006479733728852102>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-3.326853599604482>, <-2.963168738146546e-19,8.54931595814368e-19,0.006479733728852102>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-3.326853599604482>, <-2.963168738146546e-19,8.54931595814368e-19,0.006479733728852102>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    