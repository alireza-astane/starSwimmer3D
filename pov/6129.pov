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
    sphere { m*<-1.5116732574857845,-0.3234482082931799,-0.9846888242200126>, 1 }        
    sphere {  m*<-0.04989731296317057,0.17877673266489746,8.895225418060557>, 1 }
    sphere {  m*<7.305454125036801,0.0898564566705401,-5.6842678719848045>, 1 }
    sphere {  m*<-3.7395928593784236,2.6798150821439473,-2.1270754364635938>, 1}
    sphere { m*<-2.8858894380721742,-2.902351783475063,-1.662251180669545>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.04989731296317057,0.17877673266489746,8.895225418060557>, <-1.5116732574857845,-0.3234482082931799,-0.9846888242200126>, 0.5 }
    cylinder { m*<7.305454125036801,0.0898564566705401,-5.6842678719848045>, <-1.5116732574857845,-0.3234482082931799,-0.9846888242200126>, 0.5}
    cylinder { m*<-3.7395928593784236,2.6798150821439473,-2.1270754364635938>, <-1.5116732574857845,-0.3234482082931799,-0.9846888242200126>, 0.5 }
    cylinder {  m*<-2.8858894380721742,-2.902351783475063,-1.662251180669545>, <-1.5116732574857845,-0.3234482082931799,-0.9846888242200126>, 0.5}

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
    sphere { m*<-1.5116732574857845,-0.3234482082931799,-0.9846888242200126>, 1 }        
    sphere {  m*<-0.04989731296317057,0.17877673266489746,8.895225418060557>, 1 }
    sphere {  m*<7.305454125036801,0.0898564566705401,-5.6842678719848045>, 1 }
    sphere {  m*<-3.7395928593784236,2.6798150821439473,-2.1270754364635938>, 1}
    sphere { m*<-2.8858894380721742,-2.902351783475063,-1.662251180669545>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.04989731296317057,0.17877673266489746,8.895225418060557>, <-1.5116732574857845,-0.3234482082931799,-0.9846888242200126>, 0.5 }
    cylinder { m*<7.305454125036801,0.0898564566705401,-5.6842678719848045>, <-1.5116732574857845,-0.3234482082931799,-0.9846888242200126>, 0.5}
    cylinder { m*<-3.7395928593784236,2.6798150821439473,-2.1270754364635938>, <-1.5116732574857845,-0.3234482082931799,-0.9846888242200126>, 0.5 }
    cylinder {  m*<-2.8858894380721742,-2.902351783475063,-1.662251180669545>, <-1.5116732574857845,-0.3234482082931799,-0.9846888242200126>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    