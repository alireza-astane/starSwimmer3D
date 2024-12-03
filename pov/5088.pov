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
    sphere { m*<-0.36848796225947805,-0.1421922454668147,-1.634840380721299>, 1 }        
    sphere {  m*<0.49323084857673755,0.28988618322686016,8.318586349615066>, 1 }
    sphere {  m*<2.9252969472690133,-0.020210306539610673,-3.13891688762373>, 1 }
    sphere {  m*<-1.9953294993011201,2.1863399022565724,-2.5998902756860347>, 1}
    sphere { m*<-1.7275422782632883,-2.701352040147325,-2.4103439905234643>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.49323084857673755,0.28988618322686016,8.318586349615066>, <-0.36848796225947805,-0.1421922454668147,-1.634840380721299>, 0.5 }
    cylinder { m*<2.9252969472690133,-0.020210306539610673,-3.13891688762373>, <-0.36848796225947805,-0.1421922454668147,-1.634840380721299>, 0.5}
    cylinder { m*<-1.9953294993011201,2.1863399022565724,-2.5998902756860347>, <-0.36848796225947805,-0.1421922454668147,-1.634840380721299>, 0.5 }
    cylinder {  m*<-1.7275422782632883,-2.701352040147325,-2.4103439905234643>, <-0.36848796225947805,-0.1421922454668147,-1.634840380721299>, 0.5}

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
    sphere { m*<-0.36848796225947805,-0.1421922454668147,-1.634840380721299>, 1 }        
    sphere {  m*<0.49323084857673755,0.28988618322686016,8.318586349615066>, 1 }
    sphere {  m*<2.9252969472690133,-0.020210306539610673,-3.13891688762373>, 1 }
    sphere {  m*<-1.9953294993011201,2.1863399022565724,-2.5998902756860347>, 1}
    sphere { m*<-1.7275422782632883,-2.701352040147325,-2.4103439905234643>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.49323084857673755,0.28988618322686016,8.318586349615066>, <-0.36848796225947805,-0.1421922454668147,-1.634840380721299>, 0.5 }
    cylinder { m*<2.9252969472690133,-0.020210306539610673,-3.13891688762373>, <-0.36848796225947805,-0.1421922454668147,-1.634840380721299>, 0.5}
    cylinder { m*<-1.9953294993011201,2.1863399022565724,-2.5998902756860347>, <-0.36848796225947805,-0.1421922454668147,-1.634840380721299>, 0.5 }
    cylinder {  m*<-1.7275422782632883,-2.701352040147325,-2.4103439905234643>, <-0.36848796225947805,-0.1421922454668147,-1.634840380721299>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    