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
    sphere { m*<0.5051163478330953,1.1627284394934607,0.16461076722127926>, 1 }        
    sphere {  m*<0.7458514525747868,1.2914385176737861,3.152165538341828>, 1 }
    sphere {  m*<3.2398247418393518,1.264762414879835,-1.0645987582299052>, 1 }
    sphere {  m*<-1.1164990120597933,3.49120238391206,-0.8093349981946913>, 1}
    sphere { m*<-3.979213724101144,-7.314254642499843,-2.4335817778713777>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7458514525747868,1.2914385176737861,3.152165538341828>, <0.5051163478330953,1.1627284394934607,0.16461076722127926>, 0.5 }
    cylinder { m*<3.2398247418393518,1.264762414879835,-1.0645987582299052>, <0.5051163478330953,1.1627284394934607,0.16461076722127926>, 0.5}
    cylinder { m*<-1.1164990120597933,3.49120238391206,-0.8093349981946913>, <0.5051163478330953,1.1627284394934607,0.16461076722127926>, 0.5 }
    cylinder {  m*<-3.979213724101144,-7.314254642499843,-2.4335817778713777>, <0.5051163478330953,1.1627284394934607,0.16461076722127926>, 0.5}

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
    sphere { m*<0.5051163478330953,1.1627284394934607,0.16461076722127926>, 1 }        
    sphere {  m*<0.7458514525747868,1.2914385176737861,3.152165538341828>, 1 }
    sphere {  m*<3.2398247418393518,1.264762414879835,-1.0645987582299052>, 1 }
    sphere {  m*<-1.1164990120597933,3.49120238391206,-0.8093349981946913>, 1}
    sphere { m*<-3.979213724101144,-7.314254642499843,-2.4335817778713777>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7458514525747868,1.2914385176737861,3.152165538341828>, <0.5051163478330953,1.1627284394934607,0.16461076722127926>, 0.5 }
    cylinder { m*<3.2398247418393518,1.264762414879835,-1.0645987582299052>, <0.5051163478330953,1.1627284394934607,0.16461076722127926>, 0.5}
    cylinder { m*<-1.1164990120597933,3.49120238391206,-0.8093349981946913>, <0.5051163478330953,1.1627284394934607,0.16461076722127926>, 0.5 }
    cylinder {  m*<-3.979213724101144,-7.314254642499843,-2.4335817778713777>, <0.5051163478330953,1.1627284394934607,0.16461076722127926>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    