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
    sphere { m*<-0.2066462619493783,-0.10570212818153363,-0.920208567668638>, 1 }        
    sphere {  m*<0.3323596375672689,0.18247973686421443,5.768926564870273>, 1 }
    sphere {  m*<2.528062132056879,-0.0036681527951594045,-2.149418093119819>, 1 }
    sphere {  m*<-1.8282616218422683,2.2227718162370653,-1.8941543330846056>, 1}
    sphere { m*<-1.5604744008044364,-2.664920126166832,-1.7046080479220327>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3323596375672689,0.18247973686421443,5.768926564870273>, <-0.2066462619493783,-0.10570212818153363,-0.920208567668638>, 0.5 }
    cylinder { m*<2.528062132056879,-0.0036681527951594045,-2.149418093119819>, <-0.2066462619493783,-0.10570212818153363,-0.920208567668638>, 0.5}
    cylinder { m*<-1.8282616218422683,2.2227718162370653,-1.8941543330846056>, <-0.2066462619493783,-0.10570212818153363,-0.920208567668638>, 0.5 }
    cylinder {  m*<-1.5604744008044364,-2.664920126166832,-1.7046080479220327>, <-0.2066462619493783,-0.10570212818153363,-0.920208567668638>, 0.5}

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
    sphere { m*<-0.2066462619493783,-0.10570212818153363,-0.920208567668638>, 1 }        
    sphere {  m*<0.3323596375672689,0.18247973686421443,5.768926564870273>, 1 }
    sphere {  m*<2.528062132056879,-0.0036681527951594045,-2.149418093119819>, 1 }
    sphere {  m*<-1.8282616218422683,2.2227718162370653,-1.8941543330846056>, 1}
    sphere { m*<-1.5604744008044364,-2.664920126166832,-1.7046080479220327>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3323596375672689,0.18247973686421443,5.768926564870273>, <-0.2066462619493783,-0.10570212818153363,-0.920208567668638>, 0.5 }
    cylinder { m*<2.528062132056879,-0.0036681527951594045,-2.149418093119819>, <-0.2066462619493783,-0.10570212818153363,-0.920208567668638>, 0.5}
    cylinder { m*<-1.8282616218422683,2.2227718162370653,-1.8941543330846056>, <-0.2066462619493783,-0.10570212818153363,-0.920208567668638>, 0.5 }
    cylinder {  m*<-1.5604744008044364,-2.664920126166832,-1.7046080479220327>, <-0.2066462619493783,-0.10570212818153363,-0.920208567668638>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    