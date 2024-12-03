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
    sphere { m*<-1.0891334935493717,-0.16871573650802563,-1.3015011216315675>, 1 }        
    sphere {  m*<0.15796871151001368,0.2828243013816676,8.610122323014537>, 1 }
    sphere {  m*<5.586396057142576,0.06620138111283436,-4.667966484648305>, 1 }
    sphere {  m*<-2.7487263342419013,2.1602957879169944,-2.207838624878648>, 1}
    sphere { m*<-2.48093911320407,-2.727396154486903,-2.0182923397160777>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.15796871151001368,0.2828243013816676,8.610122323014537>, <-1.0891334935493717,-0.16871573650802563,-1.3015011216315675>, 0.5 }
    cylinder { m*<5.586396057142576,0.06620138111283436,-4.667966484648305>, <-1.0891334935493717,-0.16871573650802563,-1.3015011216315675>, 0.5}
    cylinder { m*<-2.7487263342419013,2.1602957879169944,-2.207838624878648>, <-1.0891334935493717,-0.16871573650802563,-1.3015011216315675>, 0.5 }
    cylinder {  m*<-2.48093911320407,-2.727396154486903,-2.0182923397160777>, <-1.0891334935493717,-0.16871573650802563,-1.3015011216315675>, 0.5}

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
    sphere { m*<-1.0891334935493717,-0.16871573650802563,-1.3015011216315675>, 1 }        
    sphere {  m*<0.15796871151001368,0.2828243013816676,8.610122323014537>, 1 }
    sphere {  m*<5.586396057142576,0.06620138111283436,-4.667966484648305>, 1 }
    sphere {  m*<-2.7487263342419013,2.1602957879169944,-2.207838624878648>, 1}
    sphere { m*<-2.48093911320407,-2.727396154486903,-2.0182923397160777>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.15796871151001368,0.2828243013816676,8.610122323014537>, <-1.0891334935493717,-0.16871573650802563,-1.3015011216315675>, 0.5 }
    cylinder { m*<5.586396057142576,0.06620138111283436,-4.667966484648305>, <-1.0891334935493717,-0.16871573650802563,-1.3015011216315675>, 0.5}
    cylinder { m*<-2.7487263342419013,2.1602957879169944,-2.207838624878648>, <-1.0891334935493717,-0.16871573650802563,-1.3015011216315675>, 0.5 }
    cylinder {  m*<-2.48093911320407,-2.727396154486903,-2.0182923397160777>, <-1.0891334935493717,-0.16871573650802563,-1.3015011216315675>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    