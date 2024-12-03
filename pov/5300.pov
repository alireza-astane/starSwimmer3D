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
    sphere { m*<-0.6325895306785024,-0.15229902321389724,-1.5226713185048952>, 1 }        
    sphere {  m*<0.37927197109798244,0.28743110997762483,8.416268681016012>, 1 }
    sphere {  m*<3.9780305089093773,0.015182005752811922,-3.712581937284966>, 1 }
    sphere {  m*<-2.272445038672813,2.1763990400480195,-2.4650266667433853>, 1}
    sphere { m*<-2.004657817634981,-2.711292902355878,-2.275480381580815>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.37927197109798244,0.28743110997762483,8.416268681016012>, <-0.6325895306785024,-0.15229902321389724,-1.5226713185048952>, 0.5 }
    cylinder { m*<3.9780305089093773,0.015182005752811922,-3.712581937284966>, <-0.6325895306785024,-0.15229902321389724,-1.5226713185048952>, 0.5}
    cylinder { m*<-2.272445038672813,2.1763990400480195,-2.4650266667433853>, <-0.6325895306785024,-0.15229902321389724,-1.5226713185048952>, 0.5 }
    cylinder {  m*<-2.004657817634981,-2.711292902355878,-2.275480381580815>, <-0.6325895306785024,-0.15229902321389724,-1.5226713185048952>, 0.5}

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
    sphere { m*<-0.6325895306785024,-0.15229902321389724,-1.5226713185048952>, 1 }        
    sphere {  m*<0.37927197109798244,0.28743110997762483,8.416268681016012>, 1 }
    sphere {  m*<3.9780305089093773,0.015182005752811922,-3.712581937284966>, 1 }
    sphere {  m*<-2.272445038672813,2.1763990400480195,-2.4650266667433853>, 1}
    sphere { m*<-2.004657817634981,-2.711292902355878,-2.275480381580815>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.37927197109798244,0.28743110997762483,8.416268681016012>, <-0.6325895306785024,-0.15229902321389724,-1.5226713185048952>, 0.5 }
    cylinder { m*<3.9780305089093773,0.015182005752811922,-3.712581937284966>, <-0.6325895306785024,-0.15229902321389724,-1.5226713185048952>, 0.5}
    cylinder { m*<-2.272445038672813,2.1763990400480195,-2.4650266667433853>, <-0.6325895306785024,-0.15229902321389724,-1.5226713185048952>, 0.5 }
    cylinder {  m*<-2.004657817634981,-2.711292902355878,-2.275480381580815>, <-0.6325895306785024,-0.15229902321389724,-1.5226713185048952>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    