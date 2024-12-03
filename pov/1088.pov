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
    sphere { m*<0.12473646155540005,-4.6771594508472454e-18,1.151710529161414>, 1 }        
    sphere {  m*<0.14098956511947391,-4.13383131768133e-18,4.151666996464402>, 1 }
    sphere {  m*<8.948382292376412,1.991599458116815e-18,-2.0218417723492985>, 1 }
    sphere {  m*<-4.607299345281089,8.164965809277259,-2.1560351461852774>, 1}
    sphere { m*<-4.607299345281089,-8.164965809277259,-2.15603514618528>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.14098956511947391,-4.13383131768133e-18,4.151666996464402>, <0.12473646155540005,-4.6771594508472454e-18,1.151710529161414>, 0.5 }
    cylinder { m*<8.948382292376412,1.991599458116815e-18,-2.0218417723492985>, <0.12473646155540005,-4.6771594508472454e-18,1.151710529161414>, 0.5}
    cylinder { m*<-4.607299345281089,8.164965809277259,-2.1560351461852774>, <0.12473646155540005,-4.6771594508472454e-18,1.151710529161414>, 0.5 }
    cylinder {  m*<-4.607299345281089,-8.164965809277259,-2.15603514618528>, <0.12473646155540005,-4.6771594508472454e-18,1.151710529161414>, 0.5}

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
    sphere { m*<0.12473646155540005,-4.6771594508472454e-18,1.151710529161414>, 1 }        
    sphere {  m*<0.14098956511947391,-4.13383131768133e-18,4.151666996464402>, 1 }
    sphere {  m*<8.948382292376412,1.991599458116815e-18,-2.0218417723492985>, 1 }
    sphere {  m*<-4.607299345281089,8.164965809277259,-2.1560351461852774>, 1}
    sphere { m*<-4.607299345281089,-8.164965809277259,-2.15603514618528>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.14098956511947391,-4.13383131768133e-18,4.151666996464402>, <0.12473646155540005,-4.6771594508472454e-18,1.151710529161414>, 0.5 }
    cylinder { m*<8.948382292376412,1.991599458116815e-18,-2.0218417723492985>, <0.12473646155540005,-4.6771594508472454e-18,1.151710529161414>, 0.5}
    cylinder { m*<-4.607299345281089,8.164965809277259,-2.1560351461852774>, <0.12473646155540005,-4.6771594508472454e-18,1.151710529161414>, 0.5 }
    cylinder {  m*<-4.607299345281089,-8.164965809277259,-2.15603514618528>, <0.12473646155540005,-4.6771594508472454e-18,1.151710529161414>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    