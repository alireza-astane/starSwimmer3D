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
    sphere { m*<-0.15638721472871178,-0.07883090844820183,-0.2964870855973474>, 1 }        
    sphere {  m*<0.12591481809837846,0.07210310989792618,3.2069188093366847>, 1 }
    sphere {  m*<2.578321179277545,0.02320306693817227,-1.525696611048531>, 1 }
    sphere {  m*<-1.778002574621602,2.249643035970397,-1.2704328510133176>, 1}
    sphere { m*<-1.5102153535837701,-2.6380489064335,-1.080886565850745>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.12591481809837846,0.07210310989792618,3.2069188093366847>, <-0.15638721472871178,-0.07883090844820183,-0.2964870855973474>, 0.5 }
    cylinder { m*<2.578321179277545,0.02320306693817227,-1.525696611048531>, <-0.15638721472871178,-0.07883090844820183,-0.2964870855973474>, 0.5}
    cylinder { m*<-1.778002574621602,2.249643035970397,-1.2704328510133176>, <-0.15638721472871178,-0.07883090844820183,-0.2964870855973474>, 0.5 }
    cylinder {  m*<-1.5102153535837701,-2.6380489064335,-1.080886565850745>, <-0.15638721472871178,-0.07883090844820183,-0.2964870855973474>, 0.5}

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
    sphere { m*<-0.15638721472871178,-0.07883090844820183,-0.2964870855973474>, 1 }        
    sphere {  m*<0.12591481809837846,0.07210310989792618,3.2069188093366847>, 1 }
    sphere {  m*<2.578321179277545,0.02320306693817227,-1.525696611048531>, 1 }
    sphere {  m*<-1.778002574621602,2.249643035970397,-1.2704328510133176>, 1}
    sphere { m*<-1.5102153535837701,-2.6380489064335,-1.080886565850745>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.12591481809837846,0.07210310989792618,3.2069188093366847>, <-0.15638721472871178,-0.07883090844820183,-0.2964870855973474>, 0.5 }
    cylinder { m*<2.578321179277545,0.02320306693817227,-1.525696611048531>, <-0.15638721472871178,-0.07883090844820183,-0.2964870855973474>, 0.5}
    cylinder { m*<-1.778002574621602,2.249643035970397,-1.2704328510133176>, <-0.15638721472871178,-0.07883090844820183,-0.2964870855973474>, 0.5 }
    cylinder {  m*<-1.5102153535837701,-2.6380489064335,-1.080886565850745>, <-0.15638721472871178,-0.07883090844820183,-0.2964870855973474>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    