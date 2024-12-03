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
    sphere { m*<-1.5475897750190213,-0.18408975562421764,-1.0506659666073201>, 1 }        
    sphere {  m*<-0.09902039667231499,0.27739005845710274,8.833048733673595>, 1 }
    sphere {  m*<7.092326353309952,0.11134942137987322,-5.6302985002356625>, 1 }
    sphere {  m*<-3.223448385263254,2.1452425179301526,-1.9257067144846114>, 1}
    sphere { m*<-2.9556611642254227,-2.7424494244737447,-1.736160429322041>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.09902039667231499,0.27739005845710274,8.833048733673595>, <-1.5475897750190213,-0.18408975562421764,-1.0506659666073201>, 0.5 }
    cylinder { m*<7.092326353309952,0.11134942137987322,-5.6302985002356625>, <-1.5475897750190213,-0.18408975562421764,-1.0506659666073201>, 0.5}
    cylinder { m*<-3.223448385263254,2.1452425179301526,-1.9257067144846114>, <-1.5475897750190213,-0.18408975562421764,-1.0506659666073201>, 0.5 }
    cylinder {  m*<-2.9556611642254227,-2.7424494244737447,-1.736160429322041>, <-1.5475897750190213,-0.18408975562421764,-1.0506659666073201>, 0.5}

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
    sphere { m*<-1.5475897750190213,-0.18408975562421764,-1.0506659666073201>, 1 }        
    sphere {  m*<-0.09902039667231499,0.27739005845710274,8.833048733673595>, 1 }
    sphere {  m*<7.092326353309952,0.11134942137987322,-5.6302985002356625>, 1 }
    sphere {  m*<-3.223448385263254,2.1452425179301526,-1.9257067144846114>, 1}
    sphere { m*<-2.9556611642254227,-2.7424494244737447,-1.736160429322041>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.09902039667231499,0.27739005845710274,8.833048733673595>, <-1.5475897750190213,-0.18408975562421764,-1.0506659666073201>, 0.5 }
    cylinder { m*<7.092326353309952,0.11134942137987322,-5.6302985002356625>, <-1.5475897750190213,-0.18408975562421764,-1.0506659666073201>, 0.5}
    cylinder { m*<-3.223448385263254,2.1452425179301526,-1.9257067144846114>, <-1.5475897750190213,-0.18408975562421764,-1.0506659666073201>, 0.5 }
    cylinder {  m*<-2.9556611642254227,-2.7424494244737447,-1.736160429322041>, <-1.5475897750190213,-0.18408975562421764,-1.0506659666073201>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    