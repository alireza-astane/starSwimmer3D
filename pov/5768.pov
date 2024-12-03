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
    sphere { m*<-1.2679095369064926,-0.1748276374717308,-1.2067066240890907>, 1 }        
    sphere {  m*<0.0618745536223137,0.28081124127772045,8.693969084881864>, 1 }
    sphere {  m*<6.1819277359100555,0.08431391829795584,-5.041846746240093>, 1 }
    sphere {  m*<-2.934249329226123,2.1543096606217644,-2.1002504168044416>, 1}
    sphere { m*<-2.6664621081882918,-2.733382281782133,-1.9107041316418711>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.0618745536223137,0.28081124127772045,8.693969084881864>, <-1.2679095369064926,-0.1748276374717308,-1.2067066240890907>, 0.5 }
    cylinder { m*<6.1819277359100555,0.08431391829795584,-5.041846746240093>, <-1.2679095369064926,-0.1748276374717308,-1.2067066240890907>, 0.5}
    cylinder { m*<-2.934249329226123,2.1543096606217644,-2.1002504168044416>, <-1.2679095369064926,-0.1748276374717308,-1.2067066240890907>, 0.5 }
    cylinder {  m*<-2.6664621081882918,-2.733382281782133,-1.9107041316418711>, <-1.2679095369064926,-0.1748276374717308,-1.2067066240890907>, 0.5}

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
    sphere { m*<-1.2679095369064926,-0.1748276374717308,-1.2067066240890907>, 1 }        
    sphere {  m*<0.0618745536223137,0.28081124127772045,8.693969084881864>, 1 }
    sphere {  m*<6.1819277359100555,0.08431391829795584,-5.041846746240093>, 1 }
    sphere {  m*<-2.934249329226123,2.1543096606217644,-2.1002504168044416>, 1}
    sphere { m*<-2.6664621081882918,-2.733382281782133,-1.9107041316418711>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.0618745536223137,0.28081124127772045,8.693969084881864>, <-1.2679095369064926,-0.1748276374717308,-1.2067066240890907>, 0.5 }
    cylinder { m*<6.1819277359100555,0.08431391829795584,-5.041846746240093>, <-1.2679095369064926,-0.1748276374717308,-1.2067066240890907>, 0.5}
    cylinder { m*<-2.934249329226123,2.1543096606217644,-2.1002504168044416>, <-1.2679095369064926,-0.1748276374717308,-1.2067066240890907>, 0.5 }
    cylinder {  m*<-2.6664621081882918,-2.733382281782133,-1.9107041316418711>, <-1.2679095369064926,-0.1748276374717308,-1.2067066240890907>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    