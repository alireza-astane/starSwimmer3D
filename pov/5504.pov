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
    sphere { m*<-0.9043150536083309,-0.1622191555370328,-1.3948943394616178>, 1 }        
    sphere {  m*<0.25166377084712727,0.2847721329649461,8.52798181077638>, 1 }
    sphere {  m*<4.954473657193768,0.04655113984050446,-4.282381778450654>, 1 }
    sphere {  m*<-2.5563572565700063,2.1666631693251692,-2.3152507190242373>, 1}
    sphere { m*<-2.288570035532175,-2.7210287730787286,-2.125704433861667>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.25166377084712727,0.2847721329649461,8.52798181077638>, <-0.9043150536083309,-0.1622191555370328,-1.3948943394616178>, 0.5 }
    cylinder { m*<4.954473657193768,0.04655113984050446,-4.282381778450654>, <-0.9043150536083309,-0.1622191555370328,-1.3948943394616178>, 0.5}
    cylinder { m*<-2.5563572565700063,2.1666631693251692,-2.3152507190242373>, <-0.9043150536083309,-0.1622191555370328,-1.3948943394616178>, 0.5 }
    cylinder {  m*<-2.288570035532175,-2.7210287730787286,-2.125704433861667>, <-0.9043150536083309,-0.1622191555370328,-1.3948943394616178>, 0.5}

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
    sphere { m*<-0.9043150536083309,-0.1622191555370328,-1.3948943394616178>, 1 }        
    sphere {  m*<0.25166377084712727,0.2847721329649461,8.52798181077638>, 1 }
    sphere {  m*<4.954473657193768,0.04655113984050446,-4.282381778450654>, 1 }
    sphere {  m*<-2.5563572565700063,2.1666631693251692,-2.3152507190242373>, 1}
    sphere { m*<-2.288570035532175,-2.7210287730787286,-2.125704433861667>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.25166377084712727,0.2847721329649461,8.52798181077638>, <-0.9043150536083309,-0.1622191555370328,-1.3948943394616178>, 0.5 }
    cylinder { m*<4.954473657193768,0.04655113984050446,-4.282381778450654>, <-0.9043150536083309,-0.1622191555370328,-1.3948943394616178>, 0.5}
    cylinder { m*<-2.5563572565700063,2.1666631693251692,-2.3152507190242373>, <-0.9043150536083309,-0.1622191555370328,-1.3948943394616178>, 0.5 }
    cylinder {  m*<-2.288570035532175,-2.7210287730787286,-2.125704433861667>, <-0.9043150536083309,-0.1622191555370328,-1.3948943394616178>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    