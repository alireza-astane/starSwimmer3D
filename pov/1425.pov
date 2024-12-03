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
    sphere { m*<0.583912005039215,-4.796540932779282e-18,0.9732376910160576>, 1 }        
    sphere {  m*<0.6714737129992168,-1.0878932106583534e-18,3.971962599249667>, 1 }
    sphere {  m*<7.129395815506555,2.448471206537014e-18,-1.5583839690694858>, 1 }
    sphere {  m*<-4.2254203760829965,8.164965809277259,-2.221088923752861>, 1}
    sphere { m*<-4.2254203760829965,-8.164965809277259,-2.2210889237528635>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6714737129992168,-1.0878932106583534e-18,3.971962599249667>, <0.583912005039215,-4.796540932779282e-18,0.9732376910160576>, 0.5 }
    cylinder { m*<7.129395815506555,2.448471206537014e-18,-1.5583839690694858>, <0.583912005039215,-4.796540932779282e-18,0.9732376910160576>, 0.5}
    cylinder { m*<-4.2254203760829965,8.164965809277259,-2.221088923752861>, <0.583912005039215,-4.796540932779282e-18,0.9732376910160576>, 0.5 }
    cylinder {  m*<-4.2254203760829965,-8.164965809277259,-2.2210889237528635>, <0.583912005039215,-4.796540932779282e-18,0.9732376910160576>, 0.5}

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
    sphere { m*<0.583912005039215,-4.796540932779282e-18,0.9732376910160576>, 1 }        
    sphere {  m*<0.6714737129992168,-1.0878932106583534e-18,3.971962599249667>, 1 }
    sphere {  m*<7.129395815506555,2.448471206537014e-18,-1.5583839690694858>, 1 }
    sphere {  m*<-4.2254203760829965,8.164965809277259,-2.221088923752861>, 1}
    sphere { m*<-4.2254203760829965,-8.164965809277259,-2.2210889237528635>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6714737129992168,-1.0878932106583534e-18,3.971962599249667>, <0.583912005039215,-4.796540932779282e-18,0.9732376910160576>, 0.5 }
    cylinder { m*<7.129395815506555,2.448471206537014e-18,-1.5583839690694858>, <0.583912005039215,-4.796540932779282e-18,0.9732376910160576>, 0.5}
    cylinder { m*<-4.2254203760829965,8.164965809277259,-2.221088923752861>, <0.583912005039215,-4.796540932779282e-18,0.9732376910160576>, 0.5 }
    cylinder {  m*<-4.2254203760829965,-8.164965809277259,-2.2210889237528635>, <0.583912005039215,-4.796540932779282e-18,0.9732376910160576>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    