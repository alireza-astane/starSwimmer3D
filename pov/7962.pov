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
    sphere { m*<-0.33481838727463037,-0.24970930389264323,-0.40461423232550486>, 1 }        
    sphere {  m*<1.0843491069255302,0.7402296099872732,9.444675864709634>, 1 }
    sphere {  m*<8.452136305248318,0.45513735919501075,-5.1260015643642864>, 1 }
    sphere {  m*<-6.443826888440667,6.978218732815647,-3.6351946611826813>, 1}
    sphere { m*<-4.325816828375805,-8.941324634974327,-2.252795919958691>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0843491069255302,0.7402296099872732,9.444675864709634>, <-0.33481838727463037,-0.24970930389264323,-0.40461423232550486>, 0.5 }
    cylinder { m*<8.452136305248318,0.45513735919501075,-5.1260015643642864>, <-0.33481838727463037,-0.24970930389264323,-0.40461423232550486>, 0.5}
    cylinder { m*<-6.443826888440667,6.978218732815647,-3.6351946611826813>, <-0.33481838727463037,-0.24970930389264323,-0.40461423232550486>, 0.5 }
    cylinder {  m*<-4.325816828375805,-8.941324634974327,-2.252795919958691>, <-0.33481838727463037,-0.24970930389264323,-0.40461423232550486>, 0.5}

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
    sphere { m*<-0.33481838727463037,-0.24970930389264323,-0.40461423232550486>, 1 }        
    sphere {  m*<1.0843491069255302,0.7402296099872732,9.444675864709634>, 1 }
    sphere {  m*<8.452136305248318,0.45513735919501075,-5.1260015643642864>, 1 }
    sphere {  m*<-6.443826888440667,6.978218732815647,-3.6351946611826813>, 1}
    sphere { m*<-4.325816828375805,-8.941324634974327,-2.252795919958691>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0843491069255302,0.7402296099872732,9.444675864709634>, <-0.33481838727463037,-0.24970930389264323,-0.40461423232550486>, 0.5 }
    cylinder { m*<8.452136305248318,0.45513735919501075,-5.1260015643642864>, <-0.33481838727463037,-0.24970930389264323,-0.40461423232550486>, 0.5}
    cylinder { m*<-6.443826888440667,6.978218732815647,-3.6351946611826813>, <-0.33481838727463037,-0.24970930389264323,-0.40461423232550486>, 0.5 }
    cylinder {  m*<-4.325816828375805,-8.941324634974327,-2.252795919958691>, <-0.33481838727463037,-0.24970930389264323,-0.40461423232550486>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    