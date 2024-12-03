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
    sphere { m*<0.566666963565402,-5.954285031356988e-18,0.9805127313342712>, 1 }        
    sphere {  m*<0.6511718050252253,-2.2036421089709696e-18,3.979325219692729>, 1 }
    sphere {  m*<7.199636580887038,1.699942372738927e-18,-1.5771685639829507>, 1 }
    sphere {  m*<-4.2394065181125615,8.164965809277259,-2.218714596976607>, 1}
    sphere { m*<-4.2394065181125615,-8.164965809277259,-2.2187145969766098>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6511718050252253,-2.2036421089709696e-18,3.979325219692729>, <0.566666963565402,-5.954285031356988e-18,0.9805127313342712>, 0.5 }
    cylinder { m*<7.199636580887038,1.699942372738927e-18,-1.5771685639829507>, <0.566666963565402,-5.954285031356988e-18,0.9805127313342712>, 0.5}
    cylinder { m*<-4.2394065181125615,8.164965809277259,-2.218714596976607>, <0.566666963565402,-5.954285031356988e-18,0.9805127313342712>, 0.5 }
    cylinder {  m*<-4.2394065181125615,-8.164965809277259,-2.2187145969766098>, <0.566666963565402,-5.954285031356988e-18,0.9805127313342712>, 0.5}

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
    sphere { m*<0.566666963565402,-5.954285031356988e-18,0.9805127313342712>, 1 }        
    sphere {  m*<0.6511718050252253,-2.2036421089709696e-18,3.979325219692729>, 1 }
    sphere {  m*<7.199636580887038,1.699942372738927e-18,-1.5771685639829507>, 1 }
    sphere {  m*<-4.2394065181125615,8.164965809277259,-2.218714596976607>, 1}
    sphere { m*<-4.2394065181125615,-8.164965809277259,-2.2187145969766098>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6511718050252253,-2.2036421089709696e-18,3.979325219692729>, <0.566666963565402,-5.954285031356988e-18,0.9805127313342712>, 0.5 }
    cylinder { m*<7.199636580887038,1.699942372738927e-18,-1.5771685639829507>, <0.566666963565402,-5.954285031356988e-18,0.9805127313342712>, 0.5}
    cylinder { m*<-4.2394065181125615,8.164965809277259,-2.218714596976607>, <0.566666963565402,-5.954285031356988e-18,0.9805127313342712>, 0.5 }
    cylinder {  m*<-4.2394065181125615,-8.164965809277259,-2.2187145969766098>, <0.566666963565402,-5.954285031356988e-18,0.9805127313342712>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    