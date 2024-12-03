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
    sphere { m*<0.7819520694492442,-4.136235422543458e-18,0.8856533407205038>, 1 }        
    sphere {  m*<0.9071389294337189,-2.9042933799213033e-19,3.883044824994405>, 1 }
    sphere {  m*<6.3076597194438255,4.421257481620483e-18,-1.3317453660406267>, 1 }
    sphere {  m*<-4.067090762535479,8.164965809277259,-2.2480694693383416>, 1}
    sphere { m*<-4.067090762535479,-8.164965809277259,-2.2480694693383443>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9071389294337189,-2.9042933799213033e-19,3.883044824994405>, <0.7819520694492442,-4.136235422543458e-18,0.8856533407205038>, 0.5 }
    cylinder { m*<6.3076597194438255,4.421257481620483e-18,-1.3317453660406267>, <0.7819520694492442,-4.136235422543458e-18,0.8856533407205038>, 0.5}
    cylinder { m*<-4.067090762535479,8.164965809277259,-2.2480694693383416>, <0.7819520694492442,-4.136235422543458e-18,0.8856533407205038>, 0.5 }
    cylinder {  m*<-4.067090762535479,-8.164965809277259,-2.2480694693383443>, <0.7819520694492442,-4.136235422543458e-18,0.8856533407205038>, 0.5}

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
    sphere { m*<0.7819520694492442,-4.136235422543458e-18,0.8856533407205038>, 1 }        
    sphere {  m*<0.9071389294337189,-2.9042933799213033e-19,3.883044824994405>, 1 }
    sphere {  m*<6.3076597194438255,4.421257481620483e-18,-1.3317453660406267>, 1 }
    sphere {  m*<-4.067090762535479,8.164965809277259,-2.2480694693383416>, 1}
    sphere { m*<-4.067090762535479,-8.164965809277259,-2.2480694693383443>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9071389294337189,-2.9042933799213033e-19,3.883044824994405>, <0.7819520694492442,-4.136235422543458e-18,0.8856533407205038>, 0.5 }
    cylinder { m*<6.3076597194438255,4.421257481620483e-18,-1.3317453660406267>, <0.7819520694492442,-4.136235422543458e-18,0.8856533407205038>, 0.5}
    cylinder { m*<-4.067090762535479,8.164965809277259,-2.2480694693383416>, <0.7819520694492442,-4.136235422543458e-18,0.8856533407205038>, 0.5 }
    cylinder {  m*<-4.067090762535479,-8.164965809277259,-2.2480694693383443>, <0.7819520694492442,-4.136235422543458e-18,0.8856533407205038>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    