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
    sphere { m*<-1.5265407370109996,-0.18340428681388185,-1.062709905523429>, 1 }        
    sphere {  m*<-0.08647511162332602,0.2776592525369232,8.822267466341321>, 1 }
    sphere {  m*<7.024518211531963,0.10936014762558763,-5.585839171734118>, 1 }
    sphere {  m*<-3.201724747694505,2.145913467386883,-1.9390801024363231>, 1}
    sphere { m*<-2.9339375266566736,-2.741778475017014,-1.7495338172737525>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.08647511162332602,0.2776592525369232,8.822267466341321>, <-1.5265407370109996,-0.18340428681388185,-1.062709905523429>, 0.5 }
    cylinder { m*<7.024518211531963,0.10936014762558763,-5.585839171734118>, <-1.5265407370109996,-0.18340428681388185,-1.062709905523429>, 0.5}
    cylinder { m*<-3.201724747694505,2.145913467386883,-1.9390801024363231>, <-1.5265407370109996,-0.18340428681388185,-1.062709905523429>, 0.5 }
    cylinder {  m*<-2.9339375266566736,-2.741778475017014,-1.7495338172737525>, <-1.5265407370109996,-0.18340428681388185,-1.062709905523429>, 0.5}

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
    sphere { m*<-1.5265407370109996,-0.18340428681388185,-1.062709905523429>, 1 }        
    sphere {  m*<-0.08647511162332602,0.2776592525369232,8.822267466341321>, 1 }
    sphere {  m*<7.024518211531963,0.10936014762558763,-5.585839171734118>, 1 }
    sphere {  m*<-3.201724747694505,2.145913467386883,-1.9390801024363231>, 1}
    sphere { m*<-2.9339375266566736,-2.741778475017014,-1.7495338172737525>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.08647511162332602,0.2776592525369232,8.822267466341321>, <-1.5265407370109996,-0.18340428681388185,-1.062709905523429>, 0.5 }
    cylinder { m*<7.024518211531963,0.10936014762558763,-5.585839171734118>, <-1.5265407370109996,-0.18340428681388185,-1.062709905523429>, 0.5}
    cylinder { m*<-3.201724747694505,2.145913467386883,-1.9390801024363231>, <-1.5265407370109996,-0.18340428681388185,-1.062709905523429>, 0.5 }
    cylinder {  m*<-2.9339375266566736,-2.741778475017014,-1.7495338172737525>, <-1.5265407370109996,-0.18340428681388185,-1.062709905523429>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    