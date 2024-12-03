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
    sphere { m*<1.0594310962252902,0.3757055464876255,0.49227344146217644>, 1 }        
    sphere {  m*<1.3034597947148516,0.40538841722473273,3.4821833703780394>, 1 }
    sphere {  m*<3.796706983777386,0.4053884172247326,-0.7350988381125789>, 1 }
    sphere {  m*<-3.018356001830736,6.83717568302904,-1.9187783609888298>, 1}
    sphere { m*<-3.788745177357138,-7.886999058170008,-2.3736067016413314>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.3034597947148516,0.40538841722473273,3.4821833703780394>, <1.0594310962252902,0.3757055464876255,0.49227344146217644>, 0.5 }
    cylinder { m*<3.796706983777386,0.4053884172247326,-0.7350988381125789>, <1.0594310962252902,0.3757055464876255,0.49227344146217644>, 0.5}
    cylinder { m*<-3.018356001830736,6.83717568302904,-1.9187783609888298>, <1.0594310962252902,0.3757055464876255,0.49227344146217644>, 0.5 }
    cylinder {  m*<-3.788745177357138,-7.886999058170008,-2.3736067016413314>, <1.0594310962252902,0.3757055464876255,0.49227344146217644>, 0.5}

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
    sphere { m*<1.0594310962252902,0.3757055464876255,0.49227344146217644>, 1 }        
    sphere {  m*<1.3034597947148516,0.40538841722473273,3.4821833703780394>, 1 }
    sphere {  m*<3.796706983777386,0.4053884172247326,-0.7350988381125789>, 1 }
    sphere {  m*<-3.018356001830736,6.83717568302904,-1.9187783609888298>, 1}
    sphere { m*<-3.788745177357138,-7.886999058170008,-2.3736067016413314>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.3034597947148516,0.40538841722473273,3.4821833703780394>, <1.0594310962252902,0.3757055464876255,0.49227344146217644>, 0.5 }
    cylinder { m*<3.796706983777386,0.4053884172247326,-0.7350988381125789>, <1.0594310962252902,0.3757055464876255,0.49227344146217644>, 0.5}
    cylinder { m*<-3.018356001830736,6.83717568302904,-1.9187783609888298>, <1.0594310962252902,0.3757055464876255,0.49227344146217644>, 0.5 }
    cylinder {  m*<-3.788745177357138,-7.886999058170008,-2.3736067016413314>, <1.0594310962252902,0.3757055464876255,0.49227344146217644>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    