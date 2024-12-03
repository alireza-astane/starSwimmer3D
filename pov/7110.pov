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
    sphere { m*<-0.7569808617604155,-1.1690967428001937,-0.6001124180715424>, 1 }        
    sphere {  m*<0.6621866324397473,-0.17915782892027576,9.249177678963612>, 1 }
    sphere {  m*<8.02997383076255,-0.46425007971253807,-5.321499750110326>, 1 }
    sphere {  m*<-6.865989362926446,6.058831293908115,-3.83069284692872>, 1}
    sphere { m*<-2.304460700795335,-4.539205674685577,-1.3167310644313168>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6621866324397473,-0.17915782892027576,9.249177678963612>, <-0.7569808617604155,-1.1690967428001937,-0.6001124180715424>, 0.5 }
    cylinder { m*<8.02997383076255,-0.46425007971253807,-5.321499750110326>, <-0.7569808617604155,-1.1690967428001937,-0.6001124180715424>, 0.5}
    cylinder { m*<-6.865989362926446,6.058831293908115,-3.83069284692872>, <-0.7569808617604155,-1.1690967428001937,-0.6001124180715424>, 0.5 }
    cylinder {  m*<-2.304460700795335,-4.539205674685577,-1.3167310644313168>, <-0.7569808617604155,-1.1690967428001937,-0.6001124180715424>, 0.5}

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
    sphere { m*<-0.7569808617604155,-1.1690967428001937,-0.6001124180715424>, 1 }        
    sphere {  m*<0.6621866324397473,-0.17915782892027576,9.249177678963612>, 1 }
    sphere {  m*<8.02997383076255,-0.46425007971253807,-5.321499750110326>, 1 }
    sphere {  m*<-6.865989362926446,6.058831293908115,-3.83069284692872>, 1}
    sphere { m*<-2.304460700795335,-4.539205674685577,-1.3167310644313168>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6621866324397473,-0.17915782892027576,9.249177678963612>, <-0.7569808617604155,-1.1690967428001937,-0.6001124180715424>, 0.5 }
    cylinder { m*<8.02997383076255,-0.46425007971253807,-5.321499750110326>, <-0.7569808617604155,-1.1690967428001937,-0.6001124180715424>, 0.5}
    cylinder { m*<-6.865989362926446,6.058831293908115,-3.83069284692872>, <-0.7569808617604155,-1.1690967428001937,-0.6001124180715424>, 0.5 }
    cylinder {  m*<-2.304460700795335,-4.539205674685577,-1.3167310644313168>, <-0.7569808617604155,-1.1690967428001937,-0.6001124180715424>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    