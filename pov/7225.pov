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
    sphere { m*<-0.7046703238433621,-1.0551746052582047,-0.5758880592612532>, 1 }        
    sphere {  m*<0.7144971703568007,-0.06523569137828678,9.273402037773902>, 1 }
    sphere {  m*<8.082284368679595,-0.35032794217054986,-5.297275391300033>, 1 }
    sphere {  m*<-6.813678825009389,6.172753431450105,-3.80646848811843>, 1}
    sphere { m*<-2.5819678145656275,-5.143561983044497,-1.445241153680232>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7144971703568007,-0.06523569137828678,9.273402037773902>, <-0.7046703238433621,-1.0551746052582047,-0.5758880592612532>, 0.5 }
    cylinder { m*<8.082284368679595,-0.35032794217054986,-5.297275391300033>, <-0.7046703238433621,-1.0551746052582047,-0.5758880592612532>, 0.5}
    cylinder { m*<-6.813678825009389,6.172753431450105,-3.80646848811843>, <-0.7046703238433621,-1.0551746052582047,-0.5758880592612532>, 0.5 }
    cylinder {  m*<-2.5819678145656275,-5.143561983044497,-1.445241153680232>, <-0.7046703238433621,-1.0551746052582047,-0.5758880592612532>, 0.5}

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
    sphere { m*<-0.7046703238433621,-1.0551746052582047,-0.5758880592612532>, 1 }        
    sphere {  m*<0.7144971703568007,-0.06523569137828678,9.273402037773902>, 1 }
    sphere {  m*<8.082284368679595,-0.35032794217054986,-5.297275391300033>, 1 }
    sphere {  m*<-6.813678825009389,6.172753431450105,-3.80646848811843>, 1}
    sphere { m*<-2.5819678145656275,-5.143561983044497,-1.445241153680232>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7144971703568007,-0.06523569137828678,9.273402037773902>, <-0.7046703238433621,-1.0551746052582047,-0.5758880592612532>, 0.5 }
    cylinder { m*<8.082284368679595,-0.35032794217054986,-5.297275391300033>, <-0.7046703238433621,-1.0551746052582047,-0.5758880592612532>, 0.5}
    cylinder { m*<-6.813678825009389,6.172753431450105,-3.80646848811843>, <-0.7046703238433621,-1.0551746052582047,-0.5758880592612532>, 0.5 }
    cylinder {  m*<-2.5819678145656275,-5.143561983044497,-1.445241153680232>, <-0.7046703238433621,-1.0551746052582047,-0.5758880592612532>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    