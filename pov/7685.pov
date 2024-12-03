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
    sphere { m*<-0.4790161129388262,-0.5637437946563715,-0.4713904037282155>, 1 }        
    sphere {  m*<0.9401513812613361,0.426195119223546,9.377899693306937>, 1 }
    sphere {  m*<8.307938579584135,0.14110286843128383,-5.192777735767001>, 1 }
    sphere {  m*<-6.588024614104861,6.664184242051919,-3.701970832585393>, 1}
    sphere { m*<-3.675584210410479,-7.5252449560463015,-1.9516812903840155>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9401513812613361,0.426195119223546,9.377899693306937>, <-0.4790161129388262,-0.5637437946563715,-0.4713904037282155>, 0.5 }
    cylinder { m*<8.307938579584135,0.14110286843128383,-5.192777735767001>, <-0.4790161129388262,-0.5637437946563715,-0.4713904037282155>, 0.5}
    cylinder { m*<-6.588024614104861,6.664184242051919,-3.701970832585393>, <-0.4790161129388262,-0.5637437946563715,-0.4713904037282155>, 0.5 }
    cylinder {  m*<-3.675584210410479,-7.5252449560463015,-1.9516812903840155>, <-0.4790161129388262,-0.5637437946563715,-0.4713904037282155>, 0.5}

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
    sphere { m*<-0.4790161129388262,-0.5637437946563715,-0.4713904037282155>, 1 }        
    sphere {  m*<0.9401513812613361,0.426195119223546,9.377899693306937>, 1 }
    sphere {  m*<8.307938579584135,0.14110286843128383,-5.192777735767001>, 1 }
    sphere {  m*<-6.588024614104861,6.664184242051919,-3.701970832585393>, 1}
    sphere { m*<-3.675584210410479,-7.5252449560463015,-1.9516812903840155>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9401513812613361,0.426195119223546,9.377899693306937>, <-0.4790161129388262,-0.5637437946563715,-0.4713904037282155>, 0.5 }
    cylinder { m*<8.307938579584135,0.14110286843128383,-5.192777735767001>, <-0.4790161129388262,-0.5637437946563715,-0.4713904037282155>, 0.5}
    cylinder { m*<-6.588024614104861,6.664184242051919,-3.701970832585393>, <-0.4790161129388262,-0.5637437946563715,-0.4713904037282155>, 0.5 }
    cylinder {  m*<-3.675584210410479,-7.5252449560463015,-1.9516812903840155>, <-0.4790161129388262,-0.5637437946563715,-0.4713904037282155>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    