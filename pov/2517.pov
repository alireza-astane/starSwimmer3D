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
    sphere { m*<0.8758233277874581,0.6666479987926884,0.3837126383938507>, 1 }        
    sphere {  m*<1.119263925167862,0.7236327568805698,3.3732737264801553>, 1 }
    sphere {  m*<3.6125111142303976,0.7236327568805696,-0.8440084820104627>, 1 }
    sphere {  m*<-2.4377183758622243,5.746997982294204,-1.5754605867513154>, 1}
    sphere { m*<-3.8596114044906886,-7.683846152681228,-2.4155112543455957>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.119263925167862,0.7236327568805698,3.3732737264801553>, <0.8758233277874581,0.6666479987926884,0.3837126383938507>, 0.5 }
    cylinder { m*<3.6125111142303976,0.7236327568805696,-0.8440084820104627>, <0.8758233277874581,0.6666479987926884,0.3837126383938507>, 0.5}
    cylinder { m*<-2.4377183758622243,5.746997982294204,-1.5754605867513154>, <0.8758233277874581,0.6666479987926884,0.3837126383938507>, 0.5 }
    cylinder {  m*<-3.8596114044906886,-7.683846152681228,-2.4155112543455957>, <0.8758233277874581,0.6666479987926884,0.3837126383938507>, 0.5}

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
    sphere { m*<0.8758233277874581,0.6666479987926884,0.3837126383938507>, 1 }        
    sphere {  m*<1.119263925167862,0.7236327568805698,3.3732737264801553>, 1 }
    sphere {  m*<3.6125111142303976,0.7236327568805696,-0.8440084820104627>, 1 }
    sphere {  m*<-2.4377183758622243,5.746997982294204,-1.5754605867513154>, 1}
    sphere { m*<-3.8596114044906886,-7.683846152681228,-2.4155112543455957>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.119263925167862,0.7236327568805698,3.3732737264801553>, <0.8758233277874581,0.6666479987926884,0.3837126383938507>, 0.5 }
    cylinder { m*<3.6125111142303976,0.7236327568805696,-0.8440084820104627>, <0.8758233277874581,0.6666479987926884,0.3837126383938507>, 0.5}
    cylinder { m*<-2.4377183758622243,5.746997982294204,-1.5754605867513154>, <0.8758233277874581,0.6666479987926884,0.3837126383938507>, 0.5 }
    cylinder {  m*<-3.8596114044906886,-7.683846152681228,-2.4155112543455957>, <0.8758233277874581,0.6666479987926884,0.3837126383938507>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    