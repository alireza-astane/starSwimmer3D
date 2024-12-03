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
    sphere { m*<1.1714493748226875,0.18778683960180206,0.5585063433843992>, 1 }        
    sphere {  m*<1.4156474224951507,0.2019153168589663,3.548517050215727>, 1 }
    sphere {  m*<3.9088946115576886,0.2019153168589663,-0.6687651582748908>, 1 }
    sphere {  m*<-3.362903001900951,7.509823259757197,-2.1225022280630776>, 1}
    sphere { m*<-3.74040553513473,-8.023994931399324,-2.34502261549511>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.4156474224951507,0.2019153168589663,3.548517050215727>, <1.1714493748226875,0.18778683960180206,0.5585063433843992>, 0.5 }
    cylinder { m*<3.9088946115576886,0.2019153168589663,-0.6687651582748908>, <1.1714493748226875,0.18778683960180206,0.5585063433843992>, 0.5}
    cylinder { m*<-3.362903001900951,7.509823259757197,-2.1225022280630776>, <1.1714493748226875,0.18778683960180206,0.5585063433843992>, 0.5 }
    cylinder {  m*<-3.74040553513473,-8.023994931399324,-2.34502261549511>, <1.1714493748226875,0.18778683960180206,0.5585063433843992>, 0.5}

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
    sphere { m*<1.1714493748226875,0.18778683960180206,0.5585063433843992>, 1 }        
    sphere {  m*<1.4156474224951507,0.2019153168589663,3.548517050215727>, 1 }
    sphere {  m*<3.9088946115576886,0.2019153168589663,-0.6687651582748908>, 1 }
    sphere {  m*<-3.362903001900951,7.509823259757197,-2.1225022280630776>, 1}
    sphere { m*<-3.74040553513473,-8.023994931399324,-2.34502261549511>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.4156474224951507,0.2019153168589663,3.548517050215727>, <1.1714493748226875,0.18778683960180206,0.5585063433843992>, 0.5 }
    cylinder { m*<3.9088946115576886,0.2019153168589663,-0.6687651582748908>, <1.1714493748226875,0.18778683960180206,0.5585063433843992>, 0.5}
    cylinder { m*<-3.362903001900951,7.509823259757197,-2.1225022280630776>, <1.1714493748226875,0.18778683960180206,0.5585063433843992>, 0.5 }
    cylinder {  m*<-3.74040553513473,-8.023994931399324,-2.34502261549511>, <1.1714493748226875,0.18778683960180206,0.5585063433843992>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    